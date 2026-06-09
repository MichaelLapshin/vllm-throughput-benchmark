from typing import Any, Dict, List, Tuple
import nvtx
import time
import torch
from ctypes import c_int, c_double
import enum
import os
import threading
import sys
import signal

from vllm import SamplingParams
from vllm.v1.request import Request
from vllm.config import VllmConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.spec_decode import utils
from vllm.logger import init_logger
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.request import Request
from vllm.v1.engine import EngineCoreRequest
from abc import abstractmethod

from utils import interprocess_util

import request_batching

utils._SAMPLING_EPS = -1

logger = init_logger(__name__)

PROMPT = f"I want you to repeat the following string fifty times: This is a " \
    "person that does a thing when this event happens at some time."


PARAM_MODEL_NAME = "vllm_profiler_model_name"
PARAM_NUM_OUTPUT_TOKENS = "vllm_profiler_num_output_tokens"
PARAM_PROFILER_TYPE = "vllm_profiler_profiler_type"
PARAM_PERF_FIFO_CTL_PATH = "vllm_profiler_perf_fifo_ctl_path"
PARAM_PERF_FIFO_ACK_PATH = "vllm_profiler_perf_fifo_ack_path"

class ProfilerType(enum.Enum):
    TIME_PROFILER="time_profiler"
    VLLM_PROFILER="vllm_profiler"
    NCU_PROFILER="ncu_profiler"
    PERF_PROFILER="perf_profiler"


def set_scheduler_parameters(
    model_name: str,
    num_output_tokens: int,
    profiler: ProfilerType,
    perf_fifo_ctl_path: str,
    perf_fifo_ack_path: str,
):
    shared_model_name = interprocess_util.SharedMemoryValue(name=PARAM_MODEL_NAME, create=True, fmt="64s", init_value="<none>".encode('utf-8'))
    shared_model_name.value = model_name.encode('utf-8')

    shared_num_output_tokens = interprocess_util.SharedMemoryValue(name=PARAM_NUM_OUTPUT_TOKENS, create=True, init_value=-1)
    shared_num_output_tokens.value = num_output_tokens

    shared_profiler_type = interprocess_util.SharedMemoryValue(name=PARAM_PROFILER_TYPE, create=True, fmt="32s", init_value="<none>".encode('utf-8'))
    shared_profiler_type.value = profiler.value.encode('utf-8')

    shared_perf_fifo_ctl_path = interprocess_util.SharedMemoryValue(name=PARAM_PERF_FIFO_CTL_PATH, create=True, fmt="256s", init_value="<none>".encode('utf-8'))
    shared_perf_fifo_ctl_path.value = perf_fifo_ctl_path.encode('utf-8')

    shared_perf_fifo_ack_path = interprocess_util.SharedMemoryValue(name=PARAM_PERF_FIFO_ACK_PATH, create=True, fmt="256s", init_value="<none>".encode('utf-8'))
    shared_perf_fifo_ack_path.value = perf_fifo_ack_path.encode('utf-8')


class SchedulerBase(Scheduler):
    SPECULATIVE_CONFIG = None

    # Scheduler parameters
    _MODEL_NAME_val = None
    _NUM_OUTPUT_TOKENS_val = None
    _PROFILER_TYPE_val = None
    _PERF_FIFO_CTL_PATH_val = None
    _PERF_FIFO_ACK_PATH_val = None

    # Profiling
    PROFILE_START_DISABLE_FLAG = -100000
    PROFILE_START_COUNTDOWN = interprocess_util.SharedMemoryValue(name="vllm_profile_start_countdown", create=True, init_value=PROFILE_START_DISABLE_FLAG)
    PROFILE_EXPECTED_NUM_DECODE_STEPS = interprocess_util.SharedMemoryValue(name="vllm_profile_expected_num_decode_steps", create=True, init_value=-1)

    PROFILE_ELAPSED_TIME = interprocess_util.SharedMemoryValue(name="vllm_profile_elapsed_time", create=True, fmt="d", init_value=-1) # used for sharing results
    NVTX_PROFILE_NAME = "RequestProfile"
    VLLM_PROFILER_START_EVENT = interprocess_util.SharedMemoryEvent(name="vllm_profile_start_event", create=True)
    VLLM_PROFILER_STOP_EVENT = interprocess_util.SharedMemoryEvent(name="vllm_profile_stop_event", create=True)

    @classmethod
    def MODEL_NAME(cls) -> str:
        if SchedulerBase._MODEL_NAME_val is None:
            SchedulerBase._MODEL_NAME_val = interprocess_util.SharedMemoryValue(name=PARAM_MODEL_NAME, fmt="64s").value.rstrip(b'\x00').decode('utf-8')
        return SchedulerBase._MODEL_NAME_val

    @classmethod
    def NUM_OUTPUT_TOKENS(cls) -> int:
        if SchedulerBase._NUM_OUTPUT_TOKENS_val is None:
                SchedulerBase._NUM_OUTPUT_TOKENS_val = interprocess_util.SharedMemoryValue(name=PARAM_NUM_OUTPUT_TOKENS).value
        return SchedulerBase._NUM_OUTPUT_TOKENS_val

    @classmethod
    def PROFILER_TYPE(cls) -> ProfilerType:
        if SchedulerBase._PROFILER_TYPE_val is None:
            profiler_type = interprocess_util.SharedMemoryValue(name=PARAM_PROFILER_TYPE, fmt="32s").value.rstrip(b'\x00').decode('utf-8')
            SchedulerBase._PROFILER_TYPE_val = ProfilerType(profiler_type)
        return SchedulerBase._PROFILER_TYPE_val

    @classmethod
    def PERF_FIFO_CTL_PATH(cls) -> str:
        if SchedulerBase._PERF_FIFO_CTL_PATH_val is None:
            SchedulerBase._PERF_FIFO_CTL_PATH_val = interprocess_util.SharedMemoryValue(name=PARAM_PERF_FIFO_CTL_PATH, fmt="256s").value.rstrip(b'\x00').decode('utf-8')
        return SchedulerBase._PERF_FIFO_CTL_PATH_val

    @classmethod
    def PERF_FIFO_ACK_PATH(cls) -> str:
        if SchedulerBase._PERF_FIFO_ACK_PATH_val is None:
            SchedulerBase._PERF_FIFO_ACK_PATH_val = interprocess_util.SharedMemoryValue(name=PARAM_PERF_FIFO_ACK_PATH, fmt="256s").value.rstrip(b'\x00').decode('utf-8')
        return SchedulerBase._PERF_FIFO_ACK_PATH_val

    def __init__(self, is_speculating, vllm_config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        self.is_speculating = is_speculating
        self.num_added_requests = 0
        self.enable_predict_bonus_token = os.environ.get('ENABLE_PREDICT_BONUS_TOKEN', 'true').lower() == 'true'
        assert self.enable_predict_bonus_token == False, "Perhaps remove this if we really want bonus tokens?"

        # Profiling variables
        self.profiler_running = False
        self.nvtx_range = None
        self.profile_start_time = None

        # vLLM profiler thread
        SchedulerBase.VLLM_PROFILER_START_EVENT.clear()
        SchedulerBase.VLLM_PROFILER_STOP_EVENT.clear()

    def add_request(self, request: Request) -> None:
        self.num_added_requests += 1
        return super().add_request(request)

    @classmethod
    def _profiler_register_n_request_batch(cls, num_requests, num_profiled_decode_steps, ignore_prefill=True):
        request_batching.set_batch_count(num_requests)

        with SchedulerBase.PROFILE_EXPECTED_NUM_DECODE_STEPS.get_lock():
            SchedulerBase.PROFILE_EXPECTED_NUM_DECODE_STEPS.value = num_profiled_decode_steps

        with SchedulerBase.PROFILE_START_COUNTDOWN.get_lock():
            if ignore_prefill:
                    assert SchedulerBase.PROFILE_START_COUNTDOWN.value == SchedulerBase.PROFILE_START_DISABLE_FLAG
                    SchedulerBase.PROFILE_START_COUNTDOWN.value = 1 # ignore first output token
            else:
                with SchedulerBase.PROFILE_START_COUNTDOWN.get_lock():
                    SchedulerBase.PROFILE_START_COUNTDOWN.value = 0

    def profiler_update_draft_token_ids(self, draft_token_ids: DraftTokenIds):
        raise NotImplementedError()

    def start_profiling(self):
        logger.info(f"(Thread ID: {threading.get_ident()}) Starting profiling...")
        assert not self.profiler_running

        # Some experiments may run with a CPU-only PyTorch build.
        # Guard all CUDA calls to avoid: "Torch not compiled with CUDA enabled".
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Start actual profiling tools
        match self.PROFILER_TYPE():
            case ProfilerType.NCU_PROFILER:
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "NCU profiling requires CUDA, but torch.cuda.is_available() is False."
                    )
                logger.info(f"Starting NVTX range...")
                assert self.nvtx_range is None
                self.nvtx_range = nvtx.start_range(message=f"{SchedulerBase.NVTX_PROFILE_NAME}")
            case ProfilerType.VLLM_PROFILER:
                logger.info(f"Starting vLLM profiler...")
                # Synchronous start of vLLM profiler
                SchedulerBase.VLLM_PROFILER_START_EVENT.set()
                # Wait until set to false (indicates an acknowledgement from thread)
                time.sleep(1) # Wait until vllm profiler starts
                # assert not SchedulerBase.VLLM_PROFILER_START_EVENT.is_set()
                # while SchedulerBase.VLLM_PROFILER_START_EVENT.is_set():
                #     time.sleep(0.001)
            case ProfilerType.PERF_PROFILER:
                with open(SchedulerBase.PERF_FIFO_CTL_PATH(), "w") as fifo:
                    fifo.write("enable")
                    fifo.flush()
                with open(SchedulerBase.PERF_FIFO_ACK_PATH(), "r") as file:
                    ack_msg = file.read(5)
                    assert ack_msg == "ack\n\x00"
            case ProfilerType.TIME_PROFILER:
                pass
            case _:
                assert False, "Unknown profiler type."

        assert self.profile_start_time is None, f"Profile start time is not None ({self.profile_start_time})"
        self.profile_start_time = time.perf_counter()

        # Set internal state
        with SchedulerBase.PROFILE_START_COUNTDOWN.get_lock():
            SchedulerBase.PROFILE_START_COUNTDOWN.value = 0
        self.profiler_running = True
        
    def stop_profiling(self):
        logger.info(f"(Thread ID: {threading.get_ident()}) Stopping profiling...")
        assert self.profiler_running
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # End actual profiling tools
        assert self.profile_start_time is not None
        with SchedulerBase.PROFILE_ELAPSED_TIME.get_lock():
            SchedulerBase.PROFILE_ELAPSED_TIME.value = time.perf_counter() - self.profile_start_time
        self.profile_start_time = None

        match self.PROFILER_TYPE():
            case ProfilerType.NCU_PROFILER:
                logger.info(f"Stopping NVTX range...")
                assert self.nvtx_range is not None
                nvtx.end_range(self.nvtx_range)
                self.nvtx_range = None
            case ProfilerType.VLLM_PROFILER:
                logger.info(f"Stopping vLLM profiler...")
                # Synchronous stop of vLLM profiler
                SchedulerBase.VLLM_PROFILER_STOP_EVENT.set()
                # Wait until set to false (indicates an acknowledgement from thread)
                time.sleep(1) # Wait until vllm profiler stops
                # assert not SchedulerBase.VLLM_PROFILER_STOP_EVENT.is_set()
                # while SchedulerBase.VLLM_PROFILER_STOP_EVENT.is_set():
                #     time.sleep(0.001)
            case ProfilerType.PERF_PROFILER:
                with open(SchedulerBase.PERF_FIFO_CTL_PATH(), "w") as fifo:
                    fifo.write("disable")
                    fifo.flush()
                with open(SchedulerBase.PERF_FIFO_ACK_PATH(), "r") as file:
                    ack_msg = file.read(5)
                    assert ack_msg == "ack\n\x00"
            case ProfilerType.TIME_PROFILER:
                pass
            case _:
                assert False, "Unknown profiler type."

        # Set internal state
        with SchedulerBase.PROFILE_START_COUNTDOWN.get_lock():
            SchedulerBase.PROFILE_START_COUNTDOWN.value = SchedulerBase.PROFILE_START_DISABLE_FLAG
        self.profiler_running = False

    @classmethod
    def _profiler_get_elapsed_time_once(cls) -> float: 
        with SchedulerBase.PROFILE_ELAPSED_TIME.get_lock():
            assert SchedulerBase.PROFILE_ELAPSED_TIME.value > 0, f"Time should have elapsed ({SchedulerBase.PROFILE_ELAPSED_TIME.value})"
            elapsed_time = SchedulerBase.PROFILE_ELAPSED_TIME.value
            SchedulerBase.PROFILE_ELAPSED_TIME.value = -1
            return elapsed_time

    @classmethod
    def _start_stop_vllm_profiler(cls, llm):
        if cls.PROFILER_TYPE() == ProfilerType.VLLM_PROFILER:
            SchedulerBase.VLLM_PROFILER_START_EVENT.wait()
            llm.start_profile()
            SchedulerBase.VLLM_PROFILER_START_EVENT.clear()
            SchedulerBase.VLLM_PROFILER_STOP_EVENT.wait()
            llm.stop_profile()
            SchedulerBase.VLLM_PROFILER_STOP_EVENT.clear()

    @classmethod
    def start_benchmark(cls, llm, prompt_token_ids: List[int], calibration_output_tokens_ids: List[int]) -> float:
        """
        Send and benchmark the custom scheduler's requests.

        :param cls: Custom scheduler class
        :param llm: The LLM object
        :param calibration_output_tokens_ids: The "correct" next tokens of the request.
        :return: Time to execute the benchmark, in seconds.
        :rtype: float
        """
        assert calibration_output_tokens_ids is not None

        # Start threads to monitor when to start vllm profiler
        vllm_profile_monitor_thread = threading.Thread(target=cls._start_stop_vllm_profiler, args=(llm,))
        vllm_profile_monitor_thread.start()
        cls._send_benchmark_requests(llm, prompt_token_ids, calibration_output_tokens_ids)
        vllm_profile_monitor_thread.join()

        # Wait until the test is done
        while True:
            with SchedulerBase.PROFILE_START_COUNTDOWN.get_lock():
                if SchedulerBase.PROFILE_START_COUNTDOWN.value == SchedulerBase.PROFILE_START_DISABLE_FLAG:
                    assert not SchedulerBase.VLLM_PROFILER_START_EVENT.is_set()
                    assert not SchedulerBase.VLLM_PROFILER_STOP_EVENT.is_set()
                    break
            time.sleep(0.5)
            logger.info("Profiling has not stopped yet...")
        
        return SchedulerBase._profiler_get_elapsed_time_once()

    @classmethod
    @abstractmethod
    def _send_benchmark_requests(cls, llm, prompt_token_ids: List[int], calibration_output_tokens_ids: List[int]) -> None:
        raise NotImplementedError()
    
    def profiler_after_schedule_logic(self, scheduler_output) -> None:
        if len(scheduler_output.num_scheduled_tokens) > 0:
            # Start profiling?
            with SchedulerBase.PROFILE_START_COUNTDOWN.get_lock():
                if SchedulerBase.PROFILE_START_COUNTDOWN.value != SchedulerBase.PROFILE_START_DISABLE_FLAG:
                    if SchedulerBase.PROFILE_START_COUNTDOWN.value == 0:
                        self.start_profiling()
                    SchedulerBase.PROFILE_START_COUNTDOWN.value -= 1  
        else:
            # Stop profiling?
            with SchedulerBase.PROFILE_START_COUNTDOWN.get_lock(), SchedulerBase.PROFILE_EXPECTED_NUM_DECODE_STEPS.get_lock():
                if SchedulerBase.PROFILE_START_COUNTDOWN.value != SchedulerBase.PROFILE_START_DISABLE_FLAG:
                    assert SchedulerBase.PROFILE_START_COUNTDOWN.value >= -SchedulerBase.PROFILE_EXPECTED_NUM_DECODE_STEPS.value
                    if SchedulerBase.PROFILE_START_COUNTDOWN.value == -SchedulerBase.PROFILE_EXPECTED_NUM_DECODE_STEPS.value:
                        # profiled decoded steps are done
                        self.stop_profiling()
                        SchedulerBase.PROFILE_START_COUNTDOWN.value = SchedulerBase.PROFILE_START_DISABLE_FLAG

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        if len(scheduler_output.scheduled_new_reqs) > 0:
            logger.info(f"Running {len(scheduler_output.scheduled_new_reqs)} new request...")
        logger.info(f"Scheduled {len(scheduler_output.num_scheduled_tokens)} tokens...")

        self.profiler_after_schedule_logic(scheduler_output)

        return super()._update_after_schedule(scheduler_output)

    def has_unfinished_requests(self) -> bool:
        return super().has_unfinished_requests()

class SchedulerWithOutputCalibration(SchedulerBase):
    TOKEN_MARGIN = 30 # Generate more tokens than outputs, in case needed

    def __init__(self, is_speculating, vllm_config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(is_speculating, vllm_config, *args, **kwargs)

        # Input/output recorder
        self.prompt_token_ids = []
        self.failure_token_ids = []

        self.calibration_request_id = None
        self.calibration_output_tokens_ids = None

    def add_request(self, request: Request) -> None:
        if self.num_added_requests == 0:
            self.calibration_request_id = request.request_id
        return super().add_request(request)

    def _free_request(self, request: Request) -> Dict[str, Any] | None:
        # Ensure the calibration request succeeds
        if not self.prompt_token_ids:
            self.prompt_token_ids = request.prompt_token_ids
            self.calibration_output_tokens_ids = request.output_token_ids
            # The following failure tokens can be anything, but we simply increment token ID
            self.failure_token_ids = [t + 1 for t in self.calibration_output_tokens_ids]

            assert self.prompt_token_ids
            logger.info(f"Prefill. Size: {len(self.prompt_token_ids)}. Tokens: {self.prompt_token_ids}")
            logger.info(f"Calibration. Size: {len(self.calibration_output_tokens_ids)}. Tokens: {list(self.calibration_output_tokens_ids)}")
        
        assert self.prompt_token_ids and self.calibration_output_tokens_ids and self.failure_token_ids 

        return super()._free_request(request)
    
    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        if not self.prompt_token_ids:
            # NOTE(mlapshin): This statement is what prevents calibration tokens from
            # triggering profiling start
            return None

        return self.profiler_update_draft_token_ids(draft_token_ids)
    
    def is_calibrating(self) -> bool:
        return (len(self.waiting) == 1 and self.waiting.peek_request().request_id == self.calibration_request_id) or \
            (len(self.running) == 1 and self.running[0].request_id == self.calibration_request_id)
    
    @classmethod
    def send_calibration_request(cls, llm) -> Tuple[List[int], List[int]]:
        total_output_tokens = cls.NUM_OUTPUT_TOKENS() + cls.TOKEN_MARGIN
        logger.info(f"Calibrating for {total_output_tokens} tokens ({cls.NUM_OUTPUT_TOKENS()=}).")

        engine = llm.llm_engine
        engine.add_request(
            request_id=str(next(llm.request_counter)),
            prompt=PROMPT,
            params=SamplingParams(
                temperature=0.0,
                max_tokens=total_output_tokens,
                min_tokens=total_output_tokens,
            ),
        )

        # Generate output
        while engine.has_unfinished_requests():
            step_outputs = engine.step()

        assert len(step_outputs) == 1
        response = step_outputs[0]
        assert len(response.outputs) == 1
        input_tokens = response.prompt_token_ids
        output_tokens = response.outputs[0].token_ids
        assert len(output_tokens) == total_output_tokens, f"{len(output_tokens)} < {total_output_tokens}"
        return input_tokens, output_tokens


"""
No Speculative Decoding Scheduler classes
"""
class NoSpecDecSchedulerBase(SchedulerWithOutputCalibration):
    def __init__(self, vllm_config: VllmConfig, *args, **kwargs) -> None:
        super().__init__(is_speculating=False, vllm_config=vllm_config, *args, **kwargs)


class NoSpecDecScheduler_Sequential(NoSpecDecSchedulerBase):
    @classmethod
    def _send_benchmark_requests(cls, llm, prompt_token_ids: List[int], calibration_output_tokens_ids: List[int]):
        num_output_tokens = cls.NUM_OUTPUT_TOKENS() + 1 # To ignore the prefill token
        engine = llm.llm_engine

        # Prepare requests
        SchedulerBase._profiler_register_n_request_batch(
            num_requests=1,
            num_profiled_decode_steps=num_output_tokens - 1, # ignore prefill
        )
        engine.add_request(
            request_id=str(next(llm.request_counter)),
            prompt=PROMPT,
            params=SamplingParams(temperature=0.0, max_tokens=num_output_tokens, min_tokens=num_output_tokens),
        )

        # Prefill
        step_outputs = engine.step()

        # Decode
        assert engine.has_unfinished_requests()
        while engine.has_unfinished_requests():
            step_outputs = engine.step()

        # Validate the output
        assert len(step_outputs) == 1
        output_tokens = step_outputs[0].outputs[0].token_ids
        assert calibration_output_tokens_ids[:num_output_tokens] == output_tokens, \
                    f"({len(calibration_output_tokens_ids[:num_output_tokens])})" \
                    f"{calibration_output_tokens_ids[:num_output_tokens]} == ({len(output_tokens)}) {output_tokens}"


class NoSpecDecScheduler_Batched(NoSpecDecSchedulerBase):
    @classmethod
    def _send_benchmark_requests(cls, llm, prompt_token_ids: List[int], calibration_output_tokens_ids: List[int]) -> float:
        num_test_requests = cls.NUM_OUTPUT_TOKENS()
        engine = llm.llm_engine

        assert len(calibration_output_tokens_ids) >= num_test_requests + 1, f"{len(calibration_output_tokens_ids)=}"

        # Prepare main batch
        SchedulerBase._profiler_register_n_request_batch(
            num_requests=num_test_requests,
            num_profiled_decode_steps=1,
        )
        for i in range(num_test_requests):
            # NOTE(mlapshin): Randomize the prompt so input parameters aren't shared among requests
            unique_prompt_token_ids = [token + i + 1 for token in prompt_token_ids + calibration_output_tokens_ids[:i]]

            req_id = str(next(llm.request_counter))
            sampling_params = SamplingParams(temperature=0.0, max_tokens=2, min_tokens=2)
            req = EngineCoreRequest(
                request_id=req_id,
                prompt_token_ids=unique_prompt_token_ids,
                mm_features=None,
                sampling_params=sampling_params,
                pooling_params=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
            )

            engine.add_request(request_id=req_id, prompt=req, params=sampling_params)
        
        # Prefill
        assert engine.get_num_unfinished_requests() == num_test_requests, f"{engine.get_num_unfinished_requests()=}"
        step_outputs = engine.step()
        assert len(step_outputs) == num_test_requests, f"{len(step_outputs)=} {(num_test_requests)=}"
        assert sum(len(o.outputs[0].token_ids) for o in step_outputs) == num_test_requests, "Not prefilling all requests"

        # Decode
        logger.info("Engine step...")
        while engine.has_unfinished_requests():
            step_outputs = engine.step()

        # Validate output
        logger.info("Validating...")
        assert len(step_outputs) == num_test_requests, f"{len(step_outputs)=} == {num_test_requests=}"
        for i, res in enumerate(step_outputs):
            input_tokens = res.prompt_token_ids
            assert len(input_tokens) == len(prompt_token_ids) + i, f"{len(input_tokens)=} == {(len(prompt_token_ids) + i)=}"
            output_tokens = res.outputs[0].token_ids
            assert len(output_tokens) == 2, f"{len(output_tokens)} != 2"


class NoSpecDecScheduler_Batched_16ot(NoSpecDecSchedulerBase):
    @classmethod
    def _send_benchmark_requests(cls, llm, prompt_token_ids: List[int], calibration_output_tokens_ids: List[int]) -> float:
        NUM_DECODE_TOKENS = 16
        num_test_requests = cls.NUM_OUTPUT_TOKENS()
        engine = llm.llm_engine

        assert len(calibration_output_tokens_ids) >= num_test_requests + 1, f"{len(calibration_output_tokens_ids)=}"

        # Prepare main batch
        SchedulerBase._profiler_register_n_request_batch(
            num_requests=num_test_requests,
            num_profiled_decode_steps=NUM_DECODE_TOKENS,
        )
        for i in range(num_test_requests):
            # NOTE(mlapshin): Randomize the prompt so input parameters aren't shared among requests
            unique_prompt_token_ids = [token + i + 1 for token in prompt_token_ids + calibration_output_tokens_ids[:i]]

            req_id = str(next(llm.request_counter))
            sampling_params = SamplingParams(temperature=0.0, max_tokens=NUM_DECODE_TOKENS+1, min_tokens=NUM_DECODE_TOKENS+1)
            req = EngineCoreRequest(
                request_id=req_id,
                prompt_token_ids=unique_prompt_token_ids,
                mm_features=None,
                sampling_params=sampling_params,
                pooling_params=None,
                arrival_time=time.time(),
                lora_request=None,
                cache_salt=None,
                data_parallel_rank=None,
            )

            engine.add_request(request_id=req_id, prompt=req, params=sampling_params)
        
        # Prefill
        assert engine.get_num_unfinished_requests() == num_test_requests, f"{engine.get_num_unfinished_requests()=}"
        step_outputs = engine.step()
        assert len(step_outputs) == num_test_requests, f"{len(step_outputs)=} {(num_test_requests)=}"
        assert sum(len(o.outputs[0].token_ids) for o in step_outputs) == num_test_requests, "Not prefilling all requests"

        # Decode
        logger.info("Engine step...")
        while engine.has_unfinished_requests():
            step_outputs = engine.step()

        # Validate output
        logger.info("Validating...")
        assert len(step_outputs) == num_test_requests, f"{len(step_outputs)=} == {num_test_requests=}"
        for i, res in enumerate(step_outputs):
            input_tokens = res.prompt_token_ids
            assert len(input_tokens) == len(prompt_token_ids) + i, f"{len(input_tokens)=} == {(len(prompt_token_ids) + i)=}"
            output_tokens = res.outputs[0].token_ids
            assert len(output_tokens) == NUM_DECODE_TOKENS+1, f"{len(output_tokens)} != {NUM_DECODE_TOKENS + 1}"
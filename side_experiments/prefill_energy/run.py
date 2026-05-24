from datetime import datetime
from zoneinfo import ZoneInfo
import os
import time
import csv
import asyncio
from dataclasses import fields, asdict, dataclass

from run_constants import PROJECT_DIR
from utils import energy_util, metadata_util, hardware_util

from vllm import LLM
from vllm import SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt

RESULTS_DIR = f"{PROJECT_DIR}/side_experiments/prefill_energy/results"
MODEL = "Qwen/Qwen3-14B"
NUM_REQUESTS_TOTAL = 1024
NUM_INPUT_TOKENS = 256
NUM_OUTPUT_TOKENS = 1
USE_ASYNC_LLM = False
MAX_NUM_BATCHED_TOKENS_LIST = [
    # 256, 512, 1024, 2048, 4096, 4096*2, 4096*4, 4096*8, 4096*16
    4096*8
]
NUM_REQUESTS_PER_BATCH_LIST = [
    256, 128, 64, 32, 16, 8, 4, 2, 1
]

@dataclass
class RequestParams:
    input_tokens_per_request: int
    output_tokens_per_request: int
    num_requests_per_batch: int
    num_request_batches: int
    num_warmup_request_batches: int
    max_num_batched_tokens: int

@dataclass
class RequestData(RequestParams):
    time_start_s: float
    time_end_s: float
    joules: int

def run_energy_benchmark_LLM(
    params: RequestParams,
) -> RequestData:
    assert params.output_tokens_per_request == 1
    llm = LLM(
        model=MODEL,
        enable_prefix_caching=False,
        max_model_len=params.input_tokens_per_request + params.output_tokens_per_request,
        max_num_seqs=params.num_requests_per_batch,
        max_num_batched_tokens=params.max_num_batched_tokens,
    )
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=1,
    )
    prompt_token_ids = [i for i in range(params.input_tokens_per_request)]
    prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)
               for _ in range(params.num_requests_per_batch)]

    # Run batches
    time_start_s = None
    total_batches = params.num_warmup_request_batches + params.num_request_batches
    for run_i in range(total_batches):
        is_warmup = run_i < params.num_warmup_request_batches
        if not is_warmup and time_start_s is None:
            time_start_s = time.time()
        _ = llm.generate(prompts, sampling_params)

    assert time_start_s is not None
    time_end_s = time.time()
    joules = energy_util.get_energy_joules(time_start_s, time_end_s)

    return RequestData(
        **params.__dict__,
        time_start_s=time_start_s,
        time_end_s=time_end_s,
        joules=joules,
    )

async def run_energy_benchmark_AsyncLLMEngine(
    params: RequestParams,
) -> RequestData:
    assert params.output_tokens_per_request == 1

    engine = AsyncLLMEngine.from_engine_args(
        engine_args=AsyncEngineArgs(
            model=MODEL,
            enable_prefix_caching=False,
            max_model_len=params.input_tokens_per_request + params.output_tokens_per_request,
            max_num_seqs=params.num_requests_per_batch,
            max_num_batched_tokens=params.max_num_batched_tokens,
        )
    )

    prompt = [i for i in range(params.input_tokens_per_request)]

    time_start_s = None
    for run_i in range(params.num_warmup_request_batches + params.num_request_batches):
        is_warmup = (run_i < params.num_warmup_request_batches)
        if is_warmup and time_start_s is None:
            time_start_s = time.time()

        for batch_i in range(params.num_warmup_request_batches):
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
            )

            async def handle_request(request_id: str, prompt: str):
                final_output = None
                async for request_output in engine.generate(
                    prompt, sampling_params, request_id
                ):
                    final_output = request_output
                return final_output.outputs[0].text

            tasks = [
                handle_request(f"req-{i}", prompt) 
                for i in range(params.num_requests_per_batch)
            ]
            
            await asyncio.gather(*tasks)
    
    # Gather data
    assert time_start_s is not None
    time_end_s = time.time()
    joules = energy_util.get_energy_joules(time_start_s, time_end_s)
    return RequestData(
        **params.__dict__,
        time_start_s=time_start_s,
        time_end_s=time_end_s,
        joules=joules,
    )

if __name__ == "__main__":
    timestamp = datetime.now(ZoneInfo('America/New_York'))
    results_dir = f"{RESULTS_DIR}/{timestamp}"

    metadata_util.save_metadata(
        results_dir,
        {
            "parameters": {
                "model": MODEL,
                "USE_ASYNC_LLM": USE_ASYNC_LLM,
                "MAX_NUM_BATCHED_TOKENS_LIST": MAX_NUM_BATCHED_TOKENS_LIST,
                "NUM_REQUESTS_PER_BATCH_LIST": NUM_REQUESTS_PER_BATCH_LIST,
                "input_tokens_per_request": NUM_INPUT_TOKENS,
                "output_tokens_per_request": NUM_OUTPUT_TOKENS,
                "num_warmup_request_batches": 1,
            },
            "environment": {
                "CPU_NAME": hardware_util.get_cpu_name(),
                "GPU_NAME": hardware_util.get_gpu_name(0),
                "CPU_AFFINITY": sorted(os.sched_getaffinity(0)),
                "CPU_NUMA_NODES": {
                    int(d[4:]): open(f'/sys/devices/system/node/{d}/cpulist').read().strip()
                    for d in os.listdir('/sys/devices/system/node/')
                    if d.startswith('node')
                }
            },
        }
    )

    results = []
    for max_num_batched_tokens in MAX_NUM_BATCHED_TOKENS_LIST:
        for num_requests_per_batch in NUM_REQUESTS_PER_BATCH_LIST:
            params = RequestParams(
                    input_tokens_per_request=NUM_INPUT_TOKENS,
                    output_tokens_per_request=NUM_OUTPUT_TOKENS,
                    num_requests_per_batch=num_requests_per_batch,
                    num_request_batches=NUM_REQUESTS_TOTAL//num_requests_per_batch,
                    num_warmup_request_batches=1,
                    max_num_batched_tokens=max_num_batched_tokens
                )
            
            if USE_ASYNC_LLM:
                results.append(asyncio.run(run_energy_benchmark_AsyncLLMEngine(params)))
            else:
                results.append(run_energy_benchmark_LLM(params))

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/out.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(RequestData)])
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

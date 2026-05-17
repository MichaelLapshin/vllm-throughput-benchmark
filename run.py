import os
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import time
import asyncio
from typing import List
from dataclasses import fields, asdict
import gc

import torch
if torch.cuda.is_available():
    torch.cuda.init()
    torch.cuda.synchronize()

import run_environment
from run_environment import (
    RUN_ON_CPU, GPU_RUN_NUMBER,
    CPU_NAME, GPU_NAME
)
import run_parameters
from run_parameters import (
    PARAM_NUM_WARMUP_RUNS, PARAM_NUM_RUNS,
    PARAM_MODELS,
    PARAM_NUM_CONCURRENT_REQUESTS,
    PARAM_NUM_INPUT_TOKENS, PARAM_NUM_OUTPUT_TOKENS,
    PARAM_MAX_SAMPLE_TOKENS,
    PARAM_CPU_OMP_THREADS_BINDS
)
import run_constants
from run_constants import (
    VLLM_SAMPLING_TEMPERATURE,
    PROJECT_DIR, RESULTS_DIR,
)
from utils import hardware_util, metadata_util, energy_util
from results import RequestData

async def benchmark_vllm_instance(
    model: str,
    cpu_omp_threads_bind,
    save_results_func,
):
    if cpu_omp_threads_bind is None:
        os.environ.pop('VLLM_CPU_OMP_THREADS_BIND', None)
    else:
        os.environ["VLLM_CPU_OMP_THREADS_BIND"] = cpu_omp_threads_bind    

    assert os.environ["PYTHONPATH"]

    # Run the benchmark
    from vllm import SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.inputs import TokensPrompt
    from request_batching import set_batch_count

    try:
        engine = AsyncLLMEngine.from_engine_args(
            engine_args=AsyncEngineArgs(
                model=model,
                enable_prefix_caching=False,
                max_model_len=max(PARAM_NUM_INPUT_TOKENS) + max(PARAM_NUM_OUTPUT_TOKENS),
                max_num_seqs=max(PARAM_NUM_CONCURRENT_REQUESTS),
            )
        )
    except Exception:
        print(f"WARNING: Failed to launch model `{model}` on {CPU_NAME if RUN_ON_CPU else GPU_NAME}")
        return

    for run_i in range(PARAM_NUM_WARMUP_RUNS + PARAM_NUM_RUNS):
        print(f"=== Run {run_i + 1}/{PARAM_NUM_WARMUP_RUNS + PARAM_NUM_RUNS} ===")
        is_warmup_run = (run_i < PARAM_NUM_WARMUP_RUNS)
        for num_requests in PARAM_NUM_CONCURRENT_REQUESTS:
            print(f"Running {num_requests} requests")
            for num_input_tokens in PARAM_NUM_INPUT_TOKENS:
                print(f"Running with {num_input_tokens} input tokens")

                # Create unique prompts for local test
                prompt_token_ids_list = []
                for _ in range(num_requests):
                    prompt_token_ids_list.append(
                        [i+1 for i in range(num_input_tokens)]
                    )

                # Execute generation
                for num_output_tokens in PARAM_NUM_OUTPUT_TOKENS:
                    num_sample_tokens = num_requests * (num_input_tokens + num_output_tokens)
                    if (PARAM_MAX_SAMPLE_TOKENS > 0 and num_sample_tokens > PARAM_MAX_SAMPLE_TOKENS):
                        print(f"Skipping sample its tokens ({num_sample_tokens}) exceeed max {PARAM_MAX_SAMPLE_TOKENS}.")
                        continue

                    sampling_params = SamplingParams(
                        temperature=VLLM_SAMPLING_TEMPERATURE,
                        min_tokens=num_output_tokens,
                        max_tokens=num_output_tokens,
                    )

                    async def run_request(req_id: str, prompt_token_ids: List[int]):
                        start_time = time.perf_counter()
                        time_to_token_s = []
                        final_output = None
                        async for output in engine.generate(
                                TokensPrompt(prompt_token_ids=prompt_token_ids),
                                sampling_params, 
                                request_id=req_id
                            ):
                            this_token_time = time.perf_counter()
                            time_to_token_s.append(this_token_time - start_time)
                            final_output = output
                        
                        # extract metrics
                        assert final_output.metrics is not None
                        assert num_output_tokens == final_output.metrics.num_generation_tokens
                        return (
                            final_output.metrics.queued_ts,
                            final_output.metrics.scheduled_ts, 
                            final_output.metrics.first_token_ts,
                            final_output.metrics.last_token_ts,
                            time_to_token_s
                        )               

                    async def run_request_batch(batch_size):
                        print(f"Running {batch_size} requests with {num_output_tokens} output tokens...")
                        set_batch_count(batch_size)

                        tasks = []
                        for i in range(batch_size):
                            tasks.append(asyncio.create_task(
                                run_request(f"req_id-{i}", prompt_token_ids_list[i])
                            ))
                        results = await asyncio.gather(*tasks)
                        return results

                    # Run batch of requests
                    request_batch_uid = str(datetime.now(ZoneInfo('America/New_York')))
                    gc.collect()

                    cpu_temp_before_run = hardware_util.get_cpu_cores_avg_temp()
                    gpu_temp_before_run = hardware_util.get_gpu_temp(GPU_RUN_NUMBER) if not RUN_ON_CPU else -1
                    time_start_s = time.time()
                    
                    results = await run_request_batch(num_requests)
                    
                    time_end_s = time.time()
                    try:
                        request_batch_energy_joules = energy_util.get_energy_joules(time_start_s, time_end_s)
                    except Exception:
                        request_batch_energy_joules = -1 # dummy value
                    cpu_temp_after_run = hardware_util.get_cpu_cores_avg_temp()
                    gpu_temp_after_run = hardware_util.get_gpu_temp(GPU_RUN_NUMBER) if not RUN_ON_CPU else -1

                    # Save results to file
                    results_to_save = []
                    for queued_ts, scheduled_ts, first_token_ts, last_token_ts, time_to_token_s in results:
                        if not is_warmup_run:
                            results_to_save.append(RequestData(
                                request_batch_uid=request_batch_uid,
                                model=model,
                                cpu_omp_threads_bind=cpu_omp_threads_bind if cpu_omp_threads_bind is not None else "<no bind>",
                                num_warmup_runs=PARAM_NUM_WARMUP_RUNS,
                                num_runs=PARAM_NUM_RUNS,
                                run_num=run_i - PARAM_NUM_WARMUP_RUNS,
                                num_concurrent_requests=num_requests,
                                num_input_tokens=num_input_tokens,
                                num_output_tokens=num_output_tokens,
                                # State
                                cpu_temp_before_run=cpu_temp_before_run,
                                cpu_temp_after_run=cpu_temp_after_run,
                                gpu_temp_before_run=gpu_temp_before_run,
                                gpu_temp_after_run=gpu_temp_after_run,
                                # Metrics
                                queued_ts=queued_ts,
                                scheduled_ts=scheduled_ts,
                                first_token_ts=first_token_ts,
                                last_token_ts=last_token_ts,
                                # Output
                                time_to_token_s=time_to_token_s,
                                request_batch_energy_joules=request_batch_energy_joules,
                            ))
                    for result in results_to_save:
                        save_results_func(result)

    # Give time for the program to gracefully shutdown
    print("Done. Shutting down...")
    await asyncio.to_thread(engine.shutdown, timeout=None)
    print("Shut down.")

def save_results_func(results_dir: str):
    file_path = f"{results_dir}/data.csv"
    
    def save_results(result: RequestData):
        with open(file_path, "a") as f:
            writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(RequestData)])
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(asdict(result))
    
    return save_results

def run_benchmarking():
    timestamp = datetime.now(ZoneInfo('America/New_York'))
    results_dir = f"{RESULTS_DIR}/{timestamp}"

    # Record metadata
    metadata_util.save_metadata(results_dir, {
        "constants": {k: str(v) for k, v in vars(run_constants).items() if k.isupper()},
        "parameters": {k: str(v) for k, v in vars(run_parameters).items() if k.isupper()},
        "environment": {k: str(v) for k, v in vars(run_environment).items() if k.isupper()},
    })

    for model in PARAM_MODELS:
        if RUN_ON_CPU:
            for cpu_omp_threads_bind in PARAM_CPU_OMP_THREADS_BINDS:
                asyncio.run(benchmark_vllm_instance(
                    model=model,
                    cpu_omp_threads_bind=cpu_omp_threads_bind,
                    save_results_func=save_results_func(results_dir)
                ))
        else:
            asyncio.run(benchmark_vllm_instance(
                model=model,
                cpu_omp_threads_bind=None,
                save_results_func=save_results_func(results_dir)
            ))

if __name__ == "__main__":
    hardware_util.initialize_nvml()
    run_benchmarking()
    hardware_util.shutdown_nvml()
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import time
import asyncio
from typing import List
from dataclasses import fields, asdict

import run_parameters
from run_parameters import (
    RUN_ON_CPU,
    PARAM_NUM_WARMUP_SAMPLES, PARAM_NUM_SAMPLES,
    PARAM_MODELS,
    PARAM_NUM_CONCURRENT_REQUESTS,
    PARAM_NUM_INPUT_TOKENS, PARAM_NUM_OUTPUT_TOKENS,
    PARAM_CPU_OMP_THREADS_BINDS
)
import run_constants
from run_constants import (
    VLLM_SAMPLING_TEMPERATURE,
    GPU_NUMBER,
    PROJECT_DIR, RESULTS_DIR,
)
from utils import hardware_util, metadata_util
from results import Result


async def benchmark_vllm_instance(
    model: str,
    cpu_omp_threads_bind = None,
) -> List[Result]:
    if cpu_omp_threads_bind is None:
        os.environ.pop('VLLM_CPU_OMP_THREADS_BIND', None)
    else:
        os.environ["VLLM_CPU_OMP_THREADS_BIND"] = cpu_omp_threads_bind    

    assert os.environ["PYTHONPATH"]

    # Run the benchmark
    from vllm import SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    engine = AsyncLLMEngine.from_engine_args(
        engine_args=AsyncEngineArgs(
            model=model,
            enable_prefix_caching=False,
        )
    )

    samples: List[Result] = []
    for run_i in range(PARAM_NUM_WARMUP_SAMPLES + PARAM_NUM_SAMPLES):
        is_warmup_run = (run_i < PARAM_NUM_WARMUP_SAMPLES)
        for num_requests in PARAM_NUM_CONCURRENT_REQUESTS:
            for num_input_tokens in PARAM_NUM_INPUT_TOKENS:
                # Create unique prompts for local test
                prompt_token_ids_list = []
                for ri in range(num_requests):
                    prompt_token_ids_list.append(
                        [ri+i+1 for i in range(num_input_tokens)]
                    )

                # Execute generation
                for num_output_tokens in PARAM_NUM_OUTPUT_TOKENS:
                    sampling_params = SamplingParams(
                        temperature=VLLM_SAMPLING_TEMPERATURE,
                        min_tokens=num_output_tokens,
                        max_tokens=num_output_tokens,
                    )

                    async def run_request(req_id: str):
                        start_time = time.perf_counter()
                        time_to_token_s = []
                        final_output = None
                        async for output in engine.generate(
                                prompt_token_ids_list[0], 
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
                        tasks = []
                        for i in range(batch_size):
                            tasks.append(asyncio.create_task(run_request(f"req_id-{i}")))
                        results = await asyncio.gather(*tasks)
                        return results

                    # Run batch of requests
                    request_batch_uid = str(datetime.now(ZoneInfo('America/New_York')))
                    results = await run_request_batch(num_requests)
                    for queued_ts, scheduled_ts, first_token_ts, last_token_ts, time_to_token_s in results:
                        # Gather results
                        if not is_warmup_run:
                            samples.append(Result(
                                request_batch_uid=request_batch_uid,
                                model=model,
                                cpu_name=hardware_util.get_cpu_name(),
                                gpu_name=hardware_util.get_gpu_name(GPU_NUMBER) if not RUN_ON_CPU else "",
                                run_on_cpu=RUN_ON_CPU,
                                cpu_omp_threads_bind=cpu_omp_threads_bind if cpu_omp_threads_bind is not None else "<no bind>",
                                num_warmup_samples=PARAM_NUM_WARMUP_SAMPLES,
                                num_samples=PARAM_NUM_SAMPLES,
                                num_concurrent_requests=num_requests,
                                num_input_tokens=num_input_tokens,
                                num_output_tokens=num_output_tokens,
                                # Metrics
                                queued_ts=queued_ts,
                                scheduled_ts=scheduled_ts,
                                first_token_ts=first_token_ts,
                                last_token_ts=last_token_ts,
                                # Output
                                time_to_token_s=time_to_token_s,
                            ))

    # Give time for the program to gracefully shutdown
    del engine
    time.sleep(3)

    return samples

def run_benchmarking():
    timestamp = datetime.now(ZoneInfo('America/New_York'))
    results_dir = f"{RESULTS_DIR}/{timestamp}"

    # Record metadata
    metadata_util.save_metadata(results_dir, {
        "constants": {k: str(v) for k, v in vars(run_constants).items() if k.isupper()},
        "parameters": {k: str(v) for k, v in vars(run_parameters).items() if k.isupper()}
    })

    samples: List[Result] = []
    for model in PARAM_MODELS:
        if RUN_ON_CPU:
            for cpu_omp_threads_bind in PARAM_CPU_OMP_THREADS_BINDS:
                samples += asyncio.run(benchmark_vllm_instance(
                    model=model,
                    cpu_omp_threads_bind=cpu_omp_threads_bind,
                ))
        else:
            samples += asyncio.run(benchmark_vllm_instance(
                model=model,
                cpu_omp_threads_bind=None,
            ))

    # Save
    with open(f"{results_dir}/data.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(Result)])
        writer.writeheader()
        writer.writerows([asdict(d) for d in samples])

if __name__ == "__main__":
    run_benchmarking()

import os
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
from typing import List
from dataclasses import dataclass, fields, asdict
import sys

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
from utils import hardware_util, metadata_util, process_util
from results import Result

sys.path.insert(0, f'{PROJECT_DIR}/vllm')

def benchmark_vllm_instance(
    model: str,
    cpu_omp_threads_bind = None,
) -> List[Result]:
    if cpu_omp_threads_bind is None:
        os.environ.pop('VLLM_CPU_OMP_THREADS_BIND', None)
    else:
        os.environ["VLLM_CPU_OMP_THREADS_BIND"] = cpu_omp_threads_bind    

    from vllm import LLM, SamplingParams
    llm = LLM(
        # model=model,
        model="Qwen/Qwen3-0.6B",
        enable_prefix_caching=False,
        # scheduler_cls=scheduler,
    )

    samples: List[Result] = []
    for _ in range(PARAM_NUM_WARMUP_SAMPLES + PARAM_NUM_SAMPLES):
        for num_concurrent_requests in PARAM_NUM_CONCURRENT_REQUESTS:
            for num_input_tokens in PARAM_NUM_INPUT_TOKENS:
                # Create unique prompts for local test
                prompt_token_ids_list = []
                for ri in range(num_concurrent_requests):
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

                    outputs = llm.generate(
                        prompts=prompt_token_ids_list,
                        sampling_params=sampling_params
                    )

                    # Verify results
                    assert len(outputs) == 1
                    output = outputs[0]
                    print(output.metrics)

                    # Gather results
                    # TODO: read prefill_time_s and decode_token_times_s
                    samples.append(Result(
                        model=model,
                        cpu_name=hardware_util.get_cpu_name(),
                        gpu_name=hardware_util.get_gpu_name(GPU_NUMBER),
                        run_on_cpu=RUN_ON_CPU,
                        cpu_omp_threads_bind=process_util.thread_bind_str_to_list(cpu_omp_threads_bind)
                            if cpu_omp_threads_bind is not None else [],
                        num_warmup_samples=PARAM_NUM_WARMUP_SAMPLES,
                        num_samples=PARAM_NUM_SAMPLES,
                        num_concurrent_requests=num_concurrent_requests,
                        num_input_tokens=num_output_tokens,
                        num_output_tokens=num_output_tokens,
                        prefill_time_s=0,
                        decode_token_times_s=[],
                    ))

    return samples[PARAM_NUM_WARMUP_SAMPLES:]

def run_benchmarking():
    timestamp = datetime.now(ZoneInfo('America/New_York'))
    results_dir = f"{RESULTS_DIR}/{timestamp}"

    # Record metadata
    metadata_util.save_metadata(results_dir, {
        "constants": {k: str(v) for k, v in vars(run_constants).items() if k.isupper()},
        "parameters": {k: str(v) for k, v in vars(run_parameters).items() if k.isupper()}
    })

    samples: List[Result] = []

    # Run on local GPU
    for model in PARAM_MODELS:
        samples += benchmark_vllm_instance(
            model=model,
            cpu_omp_threads_bind=None,
        )

        # Run on local CPU
        if RUN_ON_CPU:
            for cpu_omp_threads_bind in PARAM_CPU_OMP_THREADS_BINDS:
                samples += benchmark_vllm_instance(
                    model=model,
                    cpu_omp_threads_bind=cpu_omp_threads_bind,
                )

    # Save
    with open(f"{results_dir}/data.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(Result)])
        writer.writeheader()
        writer.writerows([asdict(d) for d in samples])

if __name__ == "__main__":
    run_benchmarking()

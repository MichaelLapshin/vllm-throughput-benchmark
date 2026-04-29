import asyncio
import os
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
from multiprocessing import Process
from typing import List
from dataclasses import dataclass, fields, asdict
import sys

import run_parameters
from run_parameters import (
    PARAM_NUM_WARMUP_SAMPLES, PARAM_NUM_SAMPLES,
    PARAM_MODELS,
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

sys.path.insert(0, f'{PROJECT_DIR}/vllm')

def benchmark_vllm_instance(
    model: str,
    run_on_cpu: bool,
    cpu_omp_threads_bind = None,
) -> List[Result]:
    from vllm import LLM, SamplingParams

    if run_on_cpu:
        assert cpu_omp_threads_bind is not None
        os.environ["VLLM_CPU_OMP_THREADS_BIND"] = cpu_omp_threads_bind
    os.environ["VLLM_TARGET_DEVICE"]="cpu" if run_on_cpu else "cuda"
    
    llm = LLM(model)

    samples: List[Result] = []
    for _ in range(PARAM_NUM_WARMUP_SAMPLES + PARAM_NUM_SAMPLES):
        
        for num_input_tokens in PARAM_NUM_INPUT_TOKENS:
            prompt_token_ids = [1+i for i in range(num_input_tokens)]

            for num_output_tokens in PARAM_NUM_OUTPUT_TOKENS:
                sampling_params = SamplingParams(
                    temperature=VLLM_SAMPLING_TEMPERATURE,
                    min_tokens=num_output_tokens,
                    max_tokens=num_output_tokens,
                )

                outputs = llm.generate(
                    prompts=prompt_token_ids,
                    sampling_params=sampling_params
                )

                # Verify results
                assert len(outputs) == 1
                output = outputs[0]
                print(output.metrics)

                # Gather results


                samples.append(Result(
                    model=model,
                    cpu_name=hardware_util.get_cpu_name(),
                    gpu_name=hardware_util.get_gpu_name(GPU_NUMBER),
                    run_on_cpu=run_on_cpu,
                    cpu_omp_threads_bind=cpu_threads_bind_str,
                    num_warmup_samples=PARAM_NUM_WARMUP_SAMPLES,
                    num_samples=PARAM_NUM_SAMPLES,
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
            run_on_cpu=False,
        )

        # Run on local CPU
        # for cpu_omp_threads_bind in PARAM_CPU_OMP_THREADS_BINDS:
        #     samples += benchmark_vllm_instance(
        #         model=model,
        #         run_on_cpu=True,
        #         cpu_omp_threads_bind=cpu_omp_threads_bind,
        #     )

    # Save
    with open(f"{results_dir}/data.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(Result)])
        writer.writeheader()
        writer.writerows([asdict(d) for d in samples])

if __name__ == "__main__":
    timestamp = datetime.now(ZoneInfo('America/New_York'))
    run_benchmarking()

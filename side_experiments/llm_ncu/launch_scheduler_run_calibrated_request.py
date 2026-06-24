import argparse
import importlib
import os

from vllm import LLM

from side_experiments.llm_ncu.speculative_vllm_schedulers import (
    ProfilerType, set_scheduler_parameters,
    NoSpecDecScheduler_Batched
)
from side_experiments.llm_ncu.constants import (
    GPU_MEMORY_UTILIZATION
)

def run_scheduler_single_request(
    model,
    base_scheduler,
    num_output_tokens: int,
    profiler_type: ProfilerType,
    perf_fifo_ctl_path: str,
    perf_fifo_ack_path: str,
):    
    set_scheduler_parameters(
        model,
        num_output_tokens,
        profiler_type,
        perf_fifo_ctl_path,
        perf_fifo_ack_path,
    )

    print(f"Running vLLM with {num_output_tokens} tokens...")

    # Add LD_PRELOAD
    if "LD_PRELOAD" not in os.environ:
        CONDA_PREFIX = os.environ["CONDA_PREFIX"]
        os.environ["LD_PRELOAD"] = f"{CONDA_PREFIX}/lib/libtcmalloc.so:{CONDA_PREFIX}/lib/libiomp5.so"
    assert "LD_PRELOAD" in os.environ, f"{os.environ=}"

    llm = LLM(
        model=model,
        dtype="auto",
        scheduler_cls=base_scheduler,
        enable_prefix_caching=False,
        speculative_config=scheduler.SPECULATIVE_CONFIG,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_chunked_prefill=True,
        max_model_len=2048,
        max_num_batched_tokens=16384,
    )

    scheduler.start_benchmark(llm)

    del llm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Model to run", required=True)
    parser.add_argument("-n", "--num-tokens", type=int, help="Number of tokens", required=True)
    parser.add_argument("-s", "--scheduler", type=str, help="Custom scheduler to use", required=True)
    parser.add_argument("-p", "--profiler", type=str, help="Profiler to use", required=True)
    parser.add_argument("-c", "--perf-fifo-ctl-path", type=str, help="Perf fifo control pipeline path")
    parser.add_argument("-a", "--perf-fifo-ack-path", type=str, help="Perf fifo control acknowledge pipeline path")
    args = parser.parse_args()

    # Fetch scheduler class
    scheduler_module = importlib.import_module("side_experiments.llm_ncu.speculative_vllm_schedulers")
    scheduler = getattr(scheduler_module, args.scheduler)

    run_scheduler_single_request(
        args.model,
        scheduler,
        args.num_tokens,
        ProfilerType(args.profiler),
        args.perf_fifo_ctl_path,
        args.perf_fifo_ack_path,
    )

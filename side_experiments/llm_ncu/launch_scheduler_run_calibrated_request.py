import argparse

from vllm import LLM

from side_experiments.llm_ncu.speculative_vllm_schedulers import (
    ProfilerType, get_parameterized_scheduler,
    SchedulerWithOutputCalibration, NoSpecDecScheduler_Batched
)
from side_experiments.llm_ncu.common_config import (
    GPU_MEMORY_UTILIZATION,
    SCHEDULERS_TO_TEST, BENCHMARK_OUTPUT_TOKENS
)

def run_scheduler_single_request(model, base_scheduler, num_output_tokens: int, profiler_type: ProfilerType):
    scheduler_cls = get_parameterized_scheduler(
        base_scheduler,
        model,
        num_output_tokens,
        profiler_type,
    )

    print(f"Running vLLM with {num_output_tokens} tokens...")

    llm = LLM(
        model=model,
        dtype="auto",
        scheduler_cls=scheduler_cls,
        enable_prefix_caching=False,
        speculative_config=scheduler.SPECULATIVE_CONFIG,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enable_chunked_prefill=True,
        max_model_len=256,
        max_num_batched_tokens=16384,
    )

    assert issubclass(scheduler, SchedulerWithOutputCalibration)
    input_tokens, calibration_output_tokens = scheduler.send_calibration_request(llm)
    scheduler.start_benchmark(llm, input_tokens, calibration_output_tokens)

    del llm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Model to run", required=True)
    parser.add_argument("-n", "--num-tokens", type=int, help="Number of tokens", required=True)
    parser.add_argument("-s", "--scheduler", type=str, help="Custom scheduler to use", required=True)
    parser.add_argument("-p", "--profiler", type=str, help="Profiler to use", required=True)
    args = parser.parse_args()

    # Fetch scheduler class
    scheduler = {s.__name__: s for s in SCHEDULERS_TO_TEST}[args.scheduler]
    assert scheduler in SCHEDULERS_TO_TEST

    run_scheduler_single_request(args.model, scheduler, args.num_tokens, ProfilerType(args.profiler))

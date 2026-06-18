import pathlib
import os
import importlib
import argparse

from side_experiments.llm_ncu.constants import EXPERIMENT_PATH

parser = argparse.ArgumentParser()
parser.add_argument(
    "--profile-gpu",
    action='store_true',
    help="Uses NCU for GPU profiling, Perf for CPU profiling.",
)
parser.add_argument(
    "--models",
    nargs="+", type=str,
    default=[
        "JackFram/llama-68m",
        "JackFram/llama-160m",
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-4B",
        "huggyllama/llama-7b",
        "huggyllama/llama-13b",
        "mistralai/Codestral-22B-v0.1",
    ],
    help="List of models to benchmark."
)
parser.add_argument(
    "--num-output-tokens",
    nargs="+", type=int,
    default=[1, 16, 256],
    help="Variants of number of output tokens to benchmark."
)
parser.add_argument(
    "--schedulers",
    nargs="+", type=str,
    default=[
        "NoSpecDecScheduler_Sequential",
        # "NoSpecDecScheduler_Batched",
        # "NoSpecDecScheduler_Batched_16ot",
    ],
    help="List of models to benchmark."
)
parser.add_argument(
    "--ncu-metrics",
    nargs="+", type=str,
    default=[
        "dram__bytes_write.sum",
        "dram__bytes_read.sum",

        # "dram__cycles_active.sum",
        # "dram__cycles_active_write.sum",
        # "dram__cycles_active_read.sum",

        # NOTE: https://forums.developer.nvidia.com/t/dram-throughput-metrics/167661
        # "gpu__dram_throughput is a breakdown metric based on dram_throughput and fbpa__throughput"

        # "sm__instruction_throughput.avg.pct_of_peak_sustained_elapsed",
        # "sm__instruction_throughput.avg.pct_of_peak_sustained_active",

        # "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        # "sm__throughput.avg.pct_of_peak_sustained_active",

        # "sm__inst_executed_pipe_tensor_v2.sum",
        # "smsp__pipe_tensor_cycles_active_v2.sum",

        # "sm__inst_executed.sum",

        # "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",

        # "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        # "dram__throughput.avg.pct_of_peak_sustained_active",
        # "dram__throughput.avg.peak_sustained",
        # "dram__throughput.avg.peak_sustained_active",
        # "dram__throughput.avg",
    ],
    help="NCU metrics to profile (requires GPU profiling to be enabled).",
)
parser.add_argument(
    "--perf-stat-runs",
    type=int,
    default=3,
    help="Number of runs for 'perf stat' profiling."
)
parser.add_argument(
    "--perf-stat-profile-metrics",
    action='store_true',
    help="Set whether to profile metrics (estimated values based on events) or not."
)
args = parser.parse_args()

PROFILE_GPU = args.profile_gpu

# vLLM Deployment
MODELS = args.models

# Benchmark
BENCHMARK_OUTPUT_TOKENS = args.num_output_tokens

# Schedulers
scheduler_module = importlib.import_module("side_experiments.llm_ncu.speculative_vllm_schedulers")
SCHEDULERS_TO_TEST = args.schedulers

# NCU Profiler
NCU_LIBRARY_DIR = None
NCU_METRIC_EXTENSIONS = ["", ".sum", ".avg", ".min", ".max"]
NCU_METRICS = args.ncu_metrics

# Perf Profiler
PERF_STAT_RUNS = args.perf_stat_runs
PERF_STAT_PROFILE_METRICS = args.perf_stat_profile_metrics
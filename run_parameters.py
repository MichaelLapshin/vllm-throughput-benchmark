"""
This file defines paramters to use for benchmarking.
"""

import run_environment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-warmup-runs",
    type=int,
    default=1,
    help="Number of runs before starting to capture data."
)
parser.add_argument(
    "--num-runs",
    type=int,
    default=3,
    help="Number of runs. Used for computing sample deviation."
)
parser.add_argument(
    "--models",
    nargs="+", type=str,
    default=[
        "Qwen/Qwen3-0.6B",
        "JackFram/llama-68m",
        "deepseek-ai/deepseek-coder-1.3b-instruct",
    ],
    help="List of models to benchmark."
)
parser.add_argument(
    "--num-concurrent-requests",
    nargs="+", type=int,
    default=[1, 2, 4] if run_environment.RUN_ON_CPU else [1, 2, 4, 8, 16, 32, 64, 128],
    help="Variants of number of concurrent requests to benchmark."
)
parser.add_argument(
    "--num-input-tokens",
    nargs="+", type=int,
    default=[1, 2, 4, 8, 16] if run_environment.RUN_ON_CPU else [1, 2, 4, 8, 16, 32, 64, 128, 256],
    help="Variants of number of input tokens to benchmark."
)
parser.add_argument(
    "--num-output-tokens",
    nargs="+", type=int,
    default=[1, 16] if run_environment.RUN_ON_CPU else [1, 16, 512],
    help="Variants of number of output tokens to benchmark."
)
parser.add_argument(
    "--cpu-omp-threads-binds",
    nargs="+", type=str,
    default=[
        "0-3",
        "0,2,4,6",
        "0-5",
        "0-7",
        "0-15",
    ],
    help="Variants of OMP threads binds to benchmark. Used only for CPU benchmarking."
)
parser.add_argument(
    "--max-sample-tokens",
    type=int,
    default=0,
    help="Maximum number of tokens to benchmark for a sample. 0 indicates no restriction. Samples exceeding (num requests)x(input len + output_len) are skipped."
)
args = parser.parse_args()

# Benchmarking
PARAM_NUM_WARMUP_RUNS = args.num_warmup_runs
PARAM_NUM_RUNS = args.num_runs

# Models
PARAM_MODELS = args.models

# Request
PARAM_NUM_CONCURRENT_REQUESTS = args.num_concurrent_requests
PARAM_NUM_INPUT_TOKENS = args.num_input_tokens
PARAM_NUM_OUTPUT_TOKENS = args.num_output_tokens
PARAM_MAX_SAMPLE_TOKENS = args.max_sample_tokens

# Hardware (this likely needs changing)
PARAM_CPU_OMP_THREADS_BINDS = args.cpu_omp_threads_binds

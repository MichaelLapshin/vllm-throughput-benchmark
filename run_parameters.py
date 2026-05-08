"""
This file defines paramters to use for benchmarking.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Conda environment
CONDA_ENV = os.environ.get("CONDA_DEFAULT_ENV")
if CONDA_ENV == "vllm_throughput_cpu":
    RUN_ON_CPU=True
elif CONDA_ENV == "vllm_throughput_gpu":
    RUN_ON_CPU=False
else:
    raise KeyError(f"Unknown conda env '{CONDA_ENV}'")

# Benchmarking
PARAM_NUM_WARMUP_SAMPLES = 1
PARAM_NUM_SAMPLES = 3

# Models
PARAM_OPEN_MODELS = [
    "Qwen/Qwen3-0.6B",
    "JackFram/llama-68m",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
]
PARAM_GATED_MODELS = [
    # "meta-llama/Llama-3.2-1B",
    # "google/t5gemma-2-270m-270m",
    # "google/t5gemma-2-1b-1b",
] if "HF_TOKEN" in os.environ else []

PARAM_MODELS = PARAM_OPEN_MODELS + PARAM_GATED_MODELS

# Request
PARAM_NUM_CONCURRENT_REQUESTS = [1, 2, 4, 8, 16] if RUN_ON_CPU else [1, 2, 4, 8, 16, 32, 64, 128]
PARAM_NUM_INPUT_TOKENS = [1, 2, 4, 8, 16, 32, 64] if RUN_ON_CPU else [1, 2, 4, 8, 16, 32, 64, 128, 256]
PARAM_NUM_OUTPUT_TOKENS = [1, 16] if RUN_ON_CPU else [1, 16, 512]

# Hardware (this likely needs changing)
PARAM_CPU_OMP_THREADS_BINDS = [
    "0-1",
    "0-3",
    "0,2,4,6",
    "0-5",
    "0-7",
    "0-15",
    "0-3,8-11"
]

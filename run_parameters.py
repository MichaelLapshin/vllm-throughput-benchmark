"""
This file defines paramters to use for benchmarking.
"""

# Benchmarking
PARAM_NUM_WARMUP_SAMPLES = 2
PARAM_NUM_SAMPLES = 7

# Generation
PARAM_MODELS = [
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-1.7B",
    "JackFram/llama-68m",
    # "meta-llama/Llama-3.2-1B",
    # "deepseek-ai/deepseek-coder-1.3b-instruct",
]

PARAM_NUM_INPUT_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128]
PARAM_NUM_OUTPUT_TOKENS = [1, 2, 4, 8, 16]

# Hardware (this likely needs changing)
PARAM_CPU_OMP_THREADS_BINDS = [
    "0",
    "0-1",
    "1-3",
    "0-7",
    "0-15",
    "0-1,8-9"
    "0-3,8-11"
]

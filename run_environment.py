"""
This files record environment variables.
"""
import os
import torch
import vllm
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

# vLLM
VLLM_VERSION = vllm.__version__

# Environment variables
ENV_VARS = dict(os.environ)

# Torch
TORCH_ENV = torch.__config__.show()
TORCH_CPU_AVX = torch.backends.cpu.get_cpu_capability()

# CPU

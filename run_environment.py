"""
This files record environment variables.
"""
import os
import torch
import vllm
from utils import hardware_util
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
CPU_NAME = hardware_util.get_cpu_name()
CPU_AFFINITY = sorted(os.sched_getaffinity(0))
CPU_NUMA_NODES = {
    int(d[4:]): open(f'/sys/devices/system/node/{d}/cpulist').read().strip()
    for d in os.listdir('/sys/devices/system/node/')
    if d.startswith('node')
}

# GPU
GPU_RUN_NUMBER = 0
GPU_NAME = "<None>" if RUN_ON_CPU else hardware_util.get_gpu_name(GPU_RUN_NUMBER)
SYSTEM_GPU_COUNT = hardware_util.get_gpu_count()
SYSTEM_GPUS = hardware_util.get_gpu_names()
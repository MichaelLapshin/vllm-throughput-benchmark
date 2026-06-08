import pathlib
import os

from side_experiments.llm_ncu.speculative_vllm_schedulers import (
    NoSpecDecScheduler_Sequential, NoSpecDecScheduler_Batched, NoSpecDecScheduler_Batched_16ot,
)
from side_experiments.llm_ncu.constants import EXPERIMENT_PATH

# vLLM environment variables
os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
os.environ["VLLM_LOGGING_CONFIG_PATH"] = f"{EXPERIMENT_PATH}/vllm_logging.json"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1" # Switch between `InprocClient` and `MPClient`
# os.environ["VLLM_DEBUG_DUMP_PATH"] = f"{EXPERIMENT_PATH}/temp_cuda_graph_dump" # Dump CUDA Graphs?

# Benchmark environment variables
ENABLE_PREDICT_BONUS_TOKEN = False
os.environ["ENABLE_PREDICT_BONUS_TOKEN"] = "true" if ENABLE_PREDICT_BONUS_TOKEN else "false"
print(f"ENABLE_PREDICT_BONUS_TOKEN={ENABLE_PREDICT_BONUS_TOKEN}")

PROFILE_GPU = False # Uses NCU for GPU profiling, Perf for CPU profiling

# vLLM Deployment
MODELS = [
    "JackFram/llama-68m",
    # "JackFram/llama-160m",
    # "Qwen/Qwen3-0.6B",
    # "Qwen/Qwen3-4B",
    # "huggyllama/llama-7b",
    # "huggyllama/llama-13b",
    # "mistralai/Codestral-22B-v0.1",
]

GPU_MEMORY_UTILIZATION=0.97

# Benchmark
BENCHMARK_OUTPUT_TOKENS = [1]

# Schedulers
SCHEDULERS_TO_TEST = [
    NoSpecDecScheduler_Sequential,
    # NoSpecDecScheduler_Batched,
    # NoSpecDecScheduler_Batched_16ot,
] # type: ignore

# NCU Profiler
NCU_LIBRARY_DIR = None
# NCU_LIBRARY_DIR = "/usr/local/cuda/include"
NCU_METRIC_EXTENSIONS = ["", ".sum", ".avg", ".min", ".max"]

NCU_METRICS = [
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
]

import pathlib
import os

from side_experiments.llm_ncu.speculative_vllm_schedulers import (
    NoSpecDecScheduler_Sequential, NoSpecDecScheduler_Batched, NoSpecDecScheduler_Batched_16ot,
)

# Paths
EXPERIMENT_PATH = f"{pathlib.Path(__file__).parent.resolve()}"

RESULTS_PATH = f"{EXPERIMENT_PATH}/results"
os.makedirs(RESULTS_PATH, exist_ok=True)

PLOTS_PATH = f"{EXPERIMENT_PATH}/plots"
os.makedirs(PLOTS_PATH, exist_ok=True)

# Labels
SCHEDULER_LABELS = {
    NoSpecDecScheduler_Sequential.__name__: "Sequential Decoding (N tokens x 1 request)",
    NoSpecDecScheduler_Batched.__name__: "Batched Decoding (1 token x N requests)",
    NoSpecDecScheduler_Batched_16ot.__name__: "Batched Decoding (16 tokens x N requests)"
}

SCHEDULER_COLOURS = {
    NoSpecDecScheduler_Sequential.__name__: "navy",
    NoSpecDecScheduler_Batched.__name__: "orange",
    NoSpecDecScheduler_Batched_16ot.__name__: "green",
}

# vLLM environment variables
os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
os.environ["VLLM_LOGGING_CONFIG_PATH"] = f"{EXPERIMENT_PATH}/vllm_logging.json"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1" # Switch between `InprocClient` and `MPClient`

# vLLM constant parameters
GPU_MEMORY_UTILIZATION=0.97

# Benchmark environment variables
ENABLE_PREDICT_BONUS_TOKEN = False
os.environ["ENABLE_PREDICT_BONUS_TOKEN"] = "true" if ENABLE_PREDICT_BONUS_TOKEN else "false"

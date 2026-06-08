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
    NoSpecDecScheduler_Sequential: "Sequential Decoding (N tokens x 1 request)",
    NoSpecDecScheduler_Batched: "Batched Decoding (1 token x N requests)",
    NoSpecDecScheduler_Batched_16ot: "Batched Decoding (16 tokens x N requests)"
}

SCHEDULER_COLOURS = {
    NoSpecDecScheduler_Sequential: "navy",
    NoSpecDecScheduler_Batched: "orange",
    NoSpecDecScheduler_Batched_16ot: "green",
}

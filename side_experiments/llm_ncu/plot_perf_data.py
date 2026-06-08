import os
import argparse
import ast
from dataclasses import dataclass
from typing import List

from utils import metadata_util

from side_experiments.llm_ncu.constants import RESULTS_PATH

@dataclass
class PerfResultsRow:
    value: float
    unit: str
    event_name: str
    runtime_s: float
    measure_time_pct: float
    metric_value: float
    metric_unit: str

@dataclass
class PerfResults:
    rows: List[PerfResultsRow]

def load_model_perf_data() -> PerfResults:
    rows = []


    results = PerfResults(rows)
    pass

def plot_metrics():
    pass

def plot_optional_metrics():
    pass

if __name__ == "__main__":
    # Load data
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default=None, help='Results directory name.')
    args = parser.parse_args()

    if args.name is None:
        results_dirs = [d for d in os.listdir(RESULTS_PATH) if os.path.isdir(os.path.join(RESULTS_PATH, d))]
        args.name = max(results_dirs, key=lambda d: os.path.getctime(os.path.join(RESULTS_PATH, d)))

    results_dir = os.path.join(RESULTS_PATH, args.name)

    # Load metadata
    metadata = metadata_util.load_metadata(results_dir)

    # Load data
    for model in ast.literal_eval(metadata["MODELS"]):
        model_dir = f"{results_dir}/{model}"
        print(f"Plotting for model: {model}")
        plot_metrics(metadata, model_dir)

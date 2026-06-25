"""
Plot the comparison of perf results between models.
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Dict
import ast
import itertools
from pathlib import Path
from collections import defaultdict
import os
import matplotlib.pyplot as plt

from utils import metadata_util
from utils.plot_utils import get_colour_cycle, format_multisample_data, int_in_range, sort_xyz

from side_experiments.llm_ncu.constants import RESULTS_PATH, PLOTS_PATH
from side_experiments.llm_ncu.perf_results import PerfRow, PerfResults

MODEL_ORDER = [
    "JackFram/llama-68m",
    "JackFram/llama-160m",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-4B",
    "huggyllama/llama-7b",
    "Qwen/Qwen3-8B",
    "huggyllama/llama-13b",
    "mistralai/Codestral-22B-v0.1",
]

def plot_model_comparison(plots_dir, models_data, cpus):
    # Gather list of events and metrics
    metrics = set([
        row.metric_unit if row.pcnt_running is None or row.pcnt_running == 100.0 else None \
        for model, data in models_data.items()
        for row in data
    ])
    events = set([
        row.event if row.pcnt_running is None or row.pcnt_running == 100.0 else None \
        for model, data in models_data.items()
        for row in data
    ])
    print("Metrics:", metrics)
    print("Events:", events)

    # Metrics
    metrics_output_dir = f"{plots_dir}/metrics"
    os.makedirs(metrics_output_dir, exist_ok=True)
    for metric in metrics:
        if metric is None:
            continue

        color_cycle = get_colour_cycle()

        x, y = [], []
        for model, data in models_data.items():
            for row in filter(lambda r: r.metric_unit == metric and (r.pcnt_running is None or r.pcnt_running == 100), data):
                x.append(model)
                y.append(row.metric_value)
        if not x:
            continue
        x, mean, std = format_multisample_data(x, y)
        x, mean, std = sort_xyz(x, mean, std, MODEL_ORDER)

        # Plot
        plt.figure(figsize=(12, 6))
        color = next(color_cycle)
        plt.plot(x, mean, marker='o', markersize=4, color=color)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
        plt.title(f"Per CPU Metric: {metric}\nCPUs: {cpus}", pad=10)
        plt.xlabel("Model")
        plt.ylabel(f"{metric}")
        plt.ylim(bottom=0)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{metrics_output_dir}/{metric.replace("/", "-")}.png", dpi=300)
        plt.close()
    
    # Events
    events_output_dir = f"{plots_dir}/events"
    os.makedirs(events_output_dir, exist_ok=True)
    for event in events:
        if event is None:
            continue

        color_cycle = get_colour_cycle()

        x, y = [], []
        unit = None
        for model, data in models_data.items():
            for row in filter(lambda r: r.event == event, data):
                x.append(model)
                y.append(row.counter_value)
                if unit is None:
                    unit = row.unit
                assert row.unit == unit
        if not x:
            continue
        x, mean, std = format_multisample_data(x, y)
        x, mean, std = sort_xyz(x, mean, std, MODEL_ORDER)

        # Plot
        plt.figure(figsize=(12, 6))
        color = next(color_cycle)
        plt.plot(x, mean, marker='o', markersize=4, color=color)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
        plt.title(f"Per CPU Events: {event}\nCPUs: {cpus}", pad=10)
        plt.xlabel("Model")
        plt.ylabel(f"{unit}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{events_output_dir}/{event.replace("/", "-")}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--names', nargs="+", type=str, help='Results directory names.')
    parser.add_argument('-a', '--all', action="store_true", help='Include all directory names.')
    parser.add_argument('-c', '--cpus', type=str, default=None, help='CPUs to filter.', required=True)
    parser.add_argument(
        '-t', '--num-output-tokens', type=int, default=1024,
        help='Use data with the specified number of output tokens.'
    )
    args = parser.parse_args()

    if args.all:
        results_path = Path(RESULTS_PATH)
        args.names = [item.name for item in results_path.iterdir() if item.is_dir()]

    # Load model data
    models_data: Dict[str, list] = defaultdict(list)
    for name in args.names:
        results_dir = os.path.join(RESULTS_PATH, name)
        metadata = metadata_util.load_metadata(results_dir)
        # benchmark_output_tokens = ast.literal_eval(metadata["parameters"]["BENCHMARK_OUTPUT_TOKENS"])
        # assert args.num_output_tokens in benchmark_output_tokens, f"Result directory '{result_dir}' should have {args.num_output_tokens} output tokens."
        
        for model in ast.literal_eval(metadata["parameters"]["MODELS"]):
            model_dir = f"{results_dir}/{model}"
            data_paths = [
                str(file.as_posix())
                # for file in Path(model_dir).rglob(f"tokens_{args.num_output_tokens}_*.jsonl")
                for file in Path(model_dir).rglob("*.jsonl")
                if "interval" not in file.name
            ]

            data = [
                i
                for dp in data_paths
                for i in PerfResults.from_jsonl(dp).rows
                if i.cpu is not None and int_in_range(i.cpu, args.cpus)
            ]

            models_data[model] += data

    # Plot the model comparison
    timestamp = datetime.now(ZoneInfo('America/New_York'))
    plots_dir = f"{PLOTS_PATH}/{timestamp}"
    plot_model_comparison(plots_dir, models_data, args.cpus)

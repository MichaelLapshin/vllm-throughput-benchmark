import os
import argparse
import ast
from typing import List
from pathlib import Path
import statistics
import json
from collections import defaultdict
import matplotlib.pyplot as plt

from utils import metadata_util
from utils.plot_utils import get_colour_cycle, format_multisample_data, int_in_range

from side_experiments.llm_metrics.constants import RESULTS_PATH
from side_experiments.llm_metrics.perf_results import PerfRow, PerfResults


def plot_intervals(metadata, data_lists, plots_dir):
    # Metrics
    metrics_output_dir = f"{plots_dir}/metrics"
    os.makedirs(metrics_output_dir, exist_ok=True)
    for dl in data_lists:
        metric_buckets = defaultdict(list)
        for r in dl:
            metric_buckets[r.metric_unit].append(r)
        
        for metric, rows in metric_buckets.items():
            if metric is None or metric == "":
                continue

            color_cycle = get_colour_cycle()

            x, y = [], []
            for r in rows:
                x.append(r.interval)
                y.append(r.metric_value)   
            x, mean, std = format_multisample_data(x, y)
                
            # Plot
            plt.figure(figsize=(12, 6))
            color = next(color_cycle)
            plt.plot(x, mean, markersize=2, color=color)
            plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
            plt.scatter(
                [r.interval for r in rows],
                [r.metric_value for r in rows],
                c=[
                    "green" if r.metric_threshold == "good" else \
                    "red" if r.metric_threshold == "bad" else \
                    "orange" if r.metric_threshold == "nearly bad" else \
                    "yellow" if r.metric_threshold == "less good" else \
                    "skyblue" for r in rows
                ],
                s=30,
            )
            plt.title(f"{metadata["parameters"]["PERF_STAT_INTERVAL_MS"]}ms Interval Sampling of Event: {metric} (pcnt-running: {rows[0].pcnt_running})", pad=10)
            plt.xlabel(f"Profiling Interval (s)")
            plt.ylabel(f"{rows[0].metric_unit}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{metrics_output_dir}/{metric.replace("/", "-")}.png", dpi=300)
            plt.close()

    # Events
    events_output_dir = f"{plots_dir}/events"
    os.makedirs(events_output_dir, exist_ok=True)
    for dl in data_lists:
        event_buckets = defaultdict(list)
        for r in dl:
            event_buckets[r.event].append(r)
        
        for event, rows in event_buckets.items():
            if event is None or event == "":
                continue

            color_cycle = get_colour_cycle()

            x, y = [], []
            for r in rows:
                x.append(r.interval)
                y.append(r.counter_value)    
            x, mean, std = format_multisample_data(x, y)
                
            # Plot
            plt.figure(figsize=(12, 6))
            color = next(color_cycle)
            plt.plot(x, mean, markersize=2, color=color)
            plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
            plt.scatter(
                [r.interval for r in rows],
                [r.counter_value for r in rows],
                color=color,
                s=30,
            )
            plt.title(f"{metadata["parameters"]["PERF_STAT_INTERVAL_MS"]}ms Interval Sampling of Event: {event} (pcnt-running: {rows[0].pcnt_running})", pad=10)
            plt.xlabel(f"Profiling Interval (s)")
            plt.ylabel(f"{rows[0].unit}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{events_output_dir}/{event.replace("/", "-")}.png", dpi=300)
            plt.close()
        

if __name__ == "__main__":
    # Load data
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default=None, help='Results directory name.', required=True))
    parser.add_argument('-c', '--cpus', type=str, default=None, help='CPUs to filter.', required=True))
    args = parser.parse_args()

    results_dir = os.path.join(RESULTS_PATH, args.name)

    # Load metadata
    metadata = metadata_util.load_metadata(results_dir)

    # Load data
    for model in ast.literal_eval(metadata["parameters"]["MODELS"]):
        model_dir = f"{results_dir}/{model}"
        data_paths = [
            str(file.as_posix())
            for file in Path(model_dir).rglob("*.jsonl")
            if not file.name.endswith(f"-interval_{metadata["parameters"]["PERF_STAT_INTERVAL_MS"]}ms.jsonl")
        ]
        interval_data_paths = [
            str(file.as_posix())
            for file in Path(model_dir).rglob(f"*-interval_{metadata["parameters"]["PERF_STAT_INTERVAL_MS"]}ms.jsonl")
        ]

        plots_dir = f"{model_dir}/plots/{args.cpus}"
        print(f"Plotting for model: {model}")

        data = [
            list(filter(lambda r: int_in_range(r.cpu, args.cpus), PerfResults.from_jsonl(dp).rows))
            for dp in data_paths
        ]
        interval_data = [
            list(filter(lambda r: int_in_range(r.cpu, args.cpus), PerfResults.from_jsonl(dp).rows))
            for dp in interval_data_paths
        ]
        plot_intervals(metadata, interval_data, plots_dir)

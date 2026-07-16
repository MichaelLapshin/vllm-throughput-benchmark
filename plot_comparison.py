import argparse
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import csv
import sys
from dataclasses import dataclass

from run_constants import RESULTS_DIR
from results import RequestData 
from utils import plot_utils, metadata_util
from typing import Callable

csv.field_size_limit(sys.maxsize)

def plot_throughputs(
    meta1, results1: list[RequestData],
    meta2, results2: list[RequestData],
    output_dir: str,
):
    print("\nGenerating throughput comparison plots...")

    device1 = meta1["environment"]["CPU_NAME"] \
        if meta1["environment"]["RUN_ON_CPU"] == "True" \
        else meta1["environment"]["GPU_NAME"] 
    device2 = meta2["environment"]["CPU_NAME"] \
        if meta2["environment"]["RUN_ON_CPU"] == "True" \
        else meta2["environment"]["GPU_NAME"] 

    @dataclass
    class ThroughputPlot:
        title: str
        x_label: str
        x_metric_func: Callable
        y_label: str
        y_metric_func: Callable
        r_filter: Callable

    plots = [
        ThroughputPlot(
            title="Number of Input Tokens vs. Decode Throughput",
            x_label="Number of Input Tokens",
            x_metric_func=lambda r: r.num_input_tokens,
            y_label="Output Throughput (tokens per second)",
            y_metric_func=lambda r: r.num_concurrent_requests * r.num_output_tokens/(r.time_to_token_s[-1] - time_to_token_s[0]),
            r_filter=lambda r: r.num_output_tokens > 1,
            # y_metric_func=lambda r: r.num_concurrent_requests/r.time_to_token_s[0],
        ),
        ThroughputPlot(
            title="Number of Output Tokens vs. Decode Throughput",
            x_label="Number of Output Tokens",
            x_metric_func=lambda r: r.num_output_tokens,
            y_label="Decode Throughput (output tokens per second)",
            # y_metric_func=lambda r: r.num_concurrent_requests * r.num_output_tokens/r.time_to_token_s[-1],
            y_metric_func=lambda r: r.num_concurrent_requests * r.num_output_tokens/(r.time_to_token_s[-1] - r.time_to_token_s[0]),
            r_filter=lambda r: r.num_output_tokens > 1,
        ),
        ThroughputPlot(
            title="Prefill Throughput vs. Number of Input Tokens",
            x_label="Number of Input Tokens",
            x_metric_func=lambda r: r.num_input_tokens,
            y_label="Prefill Throughput (input tokens per second)",
            y_metric_func=lambda r: r.num_concurrent_requests * r.num_input_tokens / (
                # Subtract denominator by (time[1] - time[0]) so to eliminate the memory-bandwidth variable,
                # given that decode is memory-bandwidth bound, rather than compute bound.
                # r.time_to_token_s[0] - (r.time_to_token_s[1] - r.time_to_token_s[0])
                
                # Below just uses the TTFT (prefill time)
                r.time_to_token_s[0]
            ),
            r_filter=lambda r: r.num_output_tokens >= 1,
        ),
        ThroughputPlot(
            title="Prefill Throughput vs. Number of Requests",
            x_label="Number of Requests",
            x_metric_func=lambda r: r.num_concurrent_requests,
            y_label="Prefill Throughput (input tokens per second)",
            y_metric_func=lambda r: r.num_concurrent_requests * r.num_input_tokens/r.time_to_token_s[0],
            r_filter=lambda r: True,
        ),
    ]

    for p in plots:
        print(f"Plotting '{p.title}'...")

        def get_metrics(results: list[RequestData]):
            aggregated = defaultdict(list)
            for r in results:
                x_value = None
                y_value = None
                try:
                    if not p.r_filter(r):
                        continue
                    x_value = p.x_metric_func(r)
                    y_value = p.y_metric_func(r)
                    aggregated[x_value].append(y_value)
                except Exception as e:
                    print("Error:", e)
                    pass
            # Collapse lists into means for each input token count
            try:
                return {i: sum(v)/len(v) for i, v in aggregated.items()}
            except Exception as e:
                print("Error:", e)
                return {}

        metrics1 = get_metrics(results1)
        metrics2 = get_metrics(results2)

        if not metrics1 or not metrics2:
            print(f"  [Skip] Insufficient data '{p.title}' plot. ({(not metrics1)=}, {not metrics2=})")
            continue

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # --- Plot Latency on Left Y-Axis (ax1) ---
        sorted_x1 = sorted(metrics1.keys())
        y1 = [metrics1[x] for x in sorted_x1]
        line1, = ax1.plot(sorted_x1, y1, label=f'R1: {device1} Throughput\n[min={min(y1):.2f}, max={max(y1):.2f}]', marker='o', linewidth=2)

        sorted_x2 = sorted(metrics2.keys())
        y2 = [metrics2[x] for x in sorted_x2]
        line2, = ax1.plot(sorted_x2, y2, label=f'R2: {device2} Throughput\n[min={min(y2):.2f}, max={max(y2):.2f}]', marker='s', linewidth=2)

        ax1.set_xlabel(p.x_label)
        ax1.set_ylabel(p.y_label)
        ax1.grid(True, which="both", ls="-", alpha=0.5)

        # --- Plot Ratio on Right Y-Axis (ax2) ---
        ax2 = ax1.twinx()
        common_x = sorted(list(set(metrics1.keys()) & set(metrics2.keys())))
        
        if len(common_x) >= 1:
            ratios = [metrics1[x] / metrics2[x] for x in common_x]
            line3, = ax2.plot(common_x, ratios, label=f'Ratio (R1/R2)\n[min={min(ratios):.2f}, max={max(ratios):.2f}]', marker='d', color='red', linestyle='--', linewidth=2)
            ax2.set_ylabel("Throughput Ratio (Run 1 / Run 2)")
            ax2.axhline(y=1.0, color='black', linestyle=':', alpha=0.6)
            
            # Set y-axis for ratio to be symmetric around 1 if possible or just show it clearly
            ymin, ymax = min(ratios), max(ratios)
            margin = (ymax - ymin) * 0.2 if ymax != ymin else 0.5
            ax2.set_ylim(0 - margin, ymax + margin)
        else:
            print("  [Warning] Not enough common input token counts to calculate a ratio plot.")

        # Combine legends from both axes
        lines = [line1, line2]
        if len(common_x) >= 2:
            lines.append(line3)
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')

        plt.title(p.title)
        
        plot_path = os.path.join(output_dir, f"{p.title}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"  -> Saved plot to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot and compare performance metrics from two result directories.")
    parser.add_argument('results_dir1', type=str, help='The first results directory name.')
    parser.add_argument('results_dir2', type=str, help='The second results directory name.')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = f"plot_comparisons/{timestamp}"
    os.makedirs(output_base_dir, exist_ok=True)

    # --- Data Loading ---
    results1 = plot_utils.load_csv_data(args.results_dir1)
    results2 = plot_utils.load_csv_data(args.results_dir2)

    meta_run1 = metadata_util.load_metadata(args.results_dir1)
    meta_run2 = metadata_util.load_metadata(args.results_dir2)
    metadata_util.save_metadata(
        output_base_dir,
        {
            "run_1": {
                "parameters": {
                    "PARAM_MODELS": meta_run1["parameters"]["PARAM_MODELS"],
                    "PARAM_NUM_CONCURRENT_REQUESTS": meta_run1["parameters"]["PARAM_NUM_CONCURRENT_REQUESTS"],
                    "PARAM_NUM_INPUT_TOKENS": meta_run1["parameters"]["PARAM_NUM_INPUT_TOKENS"],
                    "PARAM_NUM_OUTPUT_TOKENS": meta_run1["parameters"]["PARAM_NUM_OUTPUT_TOKENS"],
                    "PARAM_NUM_WARMUP_RUNS": meta_run1["parameters"]["PARAM_NUM_WARMUP_RUNS"],
                    "PARAM_NUM_RUNS": meta_run1["parameters"]["PARAM_NUM_RUNS"],
                },
                "environment": {
                    "CONDA_ENV": meta_run1["environment"]["CONDA_ENV"],
                    "CPU": meta_run1["environment"]["CPU_NAME"],
                    "CPU_AFFINITY": meta_run1["environment"]["CPU_AFFINITY"],
                    "GPU": meta_run1["environment"]["GPU_NAME"],
                    "GPU_RUN_NUMBER": meta_run1["environment"]["GPU_RUN_NUMBER"],
                    "SYSTEM_GPUS": meta_run1["environment"]["SYSTEM_GPUS"],
                },
            },
            "run_2": {
                "parameters": {
                    "PARAM_MODELS": meta_run2["parameters"]["PARAM_MODELS"],
                    "PARAM_NUM_CONCURRENT_REQUESTS": meta_run2["parameters"]["PARAM_NUM_CONCURRENT_REQUESTS"],
                    "PARAM_NUM_INPUT_TOKENS": meta_run2["parameters"]["PARAM_NUM_INPUT_TOKENS"],
                    "PARAM_NUM_OUTPUT_TOKENS": meta_run2["parameters"]["PARAM_NUM_OUTPUT_TOKENS"],
                    "PARAM_NUM_WARMUP_RUNS": meta_run2["parameters"]["PARAM_NUM_WARMUP_RUNS"],
                    "PARAM_NUM_RUNS": meta_run2["parameters"]["PARAM_NUM_RUNS"],
                },
                "environment": {
                    "CONDA_ENV": meta_run2["environment"]["CONDA_ENV"],
                    "CPU": meta_run2["environment"]["CPU_NAME"],
                    "CPU_AFFINITY": meta_run2["environment"]["CPU_AFFINITY"],
                    "GPU": meta_run2["environment"]["GPU_NAME"],
                    "GPU_RUN_NUMBER": meta_run2["environment"]["GPU_RUN_NUMBER"],
                    "SYSTEM_GPUS": meta_run2["environment"]["SYSTEM_GPUS"],
                },
            },
        }
    )

    if not results1 and not results2:
        print("No data loaded from either directory. Exiting.")
    else:
        plot_throughputs(meta_run1, results1, meta_run2, results2, output_base_dir)
        print("\nProcess completed. Plots saved to:", output_base_dir)
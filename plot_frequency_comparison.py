import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import defaultdict
import argparse
import os
import itertools
import csv
import sys

from results import RequestData, EmissionsData
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Callable

from utils import plot_utils, metadata_util
from utils.plot_utils import (
    MARKERS, group_and_find_best_records, keep_per_request_batch,
    plot_fitted_line, get_poly_colour_no_alpha, get_colour_cycle,
    format_multisample_data
)


def plot_frequency_comparison(output_dir, metadata, results, emissions):
    cpu_name, gpu_name, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)
    
    @dataclass
    class FrequencyPlot:
        y_label_short: str
        y_label: str
        metric_fn: Callable = None
        emissions_metric_func: Callable = None

    plots = [
        FrequencyPlot(
            y_label_short="Prefill Throughput",
            y_label="Prefill Throughput (input tokens per second)",
            metric_fn=lambda r: r.num_input_tokens / r.last_token_ts,
        ),
        FrequencyPlot(
            y_label_short="System Energy",
            y_label="System Energy Consumption (joules per token)",
            metric_fn=lambda r: r.request_batch_energy_joules / r.num_input_tokens,
        ),
        FrequencyPlot(
            y_label_short="CPU Energy",
            y_label="CPU Energy Consumption (joules per input token)",
            metric_fn=lambda _: 1,
            emissions_metric_func=lambda r, e: e.cpu_energy * 3600000 / r.num_input_tokens,
        ),
        FrequencyPlot(
            y_label_short="CPU Power",
            y_label="CPU Average Power (watts)",
            metric_fn=lambda _: 1,
            emissions_metric_func=lambda _, e: e.cpu_power,
        ),
    ]

    for p in plots:
        # Keep only one request per batch for reference
        filtered_results = list(filter(lambda r: r.num_output_tokens == 1 and r.num_concurrent_requests == 1, results))

        # Group data
        groups, best_omp_thread_binds = group_and_find_best_records(
            data=filtered_results,
            group_by_fn=lambda r: r.cpu_frequency_khz,
            sub_group_by_fn=lambda r: r.num_input_tokens,
            metric_fn=p.metric_fn,
            best_attr_fn=lambda r: r.cpu_omp_threads_bind,
            minimize=True,
        )

        plt.figure(figsize=(10, 6))
        color_cycle = get_colour_cycle()

        for cpu_frequency_khz, group in groups.items():
            if not group:
                continue

            x, y = [], []
            for result in group:
                x.append(result.num_input_tokens)
                if p.emissions_metric_func:
                    y.append(p.emissions_metric_func(
                        result,
                        emissions[result.request_batch_uid]
                    ))
                else:
                    y.append(p.metric_fn(result))

            # Plot line
            x, mean, std = format_multisample_data(x, y)
            color = next(color_cycle)
            plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
            line_eq = None # plot_fitted_line(color, x, mean)
            plt.plot(x, mean, label=f"({line_eq}) " if line_eq is not None else "" \
                    f"Max Frequency: {cpu_frequency_khz // 1000} MHz", color=color, marker='o', ms=4)

        plt.title(
            f"Max Frequency vs. {p.y_label_short}\n" \
            f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
            f"Model: {model}, Output Tokens: {1}, Number of requests: {1}"
        )
        plt.xlabel('Number of Input Tokens')
        plt.ylabel(p.y_label)
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}/Frequency vs. {p.y_label_short}.png")
        plt.close()
        

if __name__ == "__main__":
    # Load data
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results-path', type=str, default=None, help='Path to a specific results directory.')
    args = parser.parse_args()

    # Fetch data
    try:
        metadata = metadata_util.load_metadata(args.results_path)
        results = plot_utils.load_csv_data(args.results_path)
        emissions = plot_utils.load_csv_emissions(args.results_path)
    except FileNotFoundError:
        print(f"File not found. Skipping...")

    # Plot
    output_dir = "plot_frequency_comparison"
    os.makedirs(output_dir, exist_ok=True)
    plot_frequency_comparison(output_dir, metadata, results, emissions)
    
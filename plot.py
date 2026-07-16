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

from run_constants import RESULTS_DIR, PLOTS_DIR
from utils import plot_utils, metadata_util
from utils.plot_utils import (
    MARKERS, group_and_find_best_records, keep_per_request_batch,
    plot_fitted_line, get_poly_colour_no_alpha, get_colour_cycle,
    format_multisample_data
)

csv.field_size_limit(sys.maxsize)

def plot_cpu_omp_threads_binds_prefill_throughput(output_dir: str, results: List[RequestData], metadata: dict):
    """
    Plot TTFT for varying input sizes.
    Graphs are separated based on the number of concurrent requests.
    """
    cpu_name, _, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)
    if not run_on_cpu:
        return

    output_dir = f"{output_dir}/cpu_omp_threads_bind_prefill_throughput"
    os.makedirs(output_dir, exist_ok=True)

    for num_concurrent_requests in set([r.num_concurrent_requests for r in results]):
        color_cycle = get_colour_cycle()
        plt.figure(figsize=(10, 6))

        # Keep results with highest execution time
        filtered_results = keep_per_request_batch(
            results, 
            lambda r: r.time_to_token_s[0],
            keep_max=True
        )

        groups = defaultdict(list)
        for result in filtered_results:
            if result.num_concurrent_requests != num_concurrent_requests:
                continue
            groups[result.cpu_omp_threads_bind].append(result)
        for key in groups:
            groups[key].sort(key=lambda result: result.num_input_tokens)
    
        for cpu_omp_threads_bind, grouped_results in groups.items():
            x, y = [], []
            for result in grouped_results:
                x.append(result.num_input_tokens)
                throughput_tps = (result.num_input_tokens * result.num_concurrent_requests) / result.time_to_token_s[0]
                y.append(throughput_tps)
            x, mean, std = format_multisample_data(x, y)
            
            color = next(color_cycle)
            plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
            line_eq = None # plot_fitted_line(color, x, mean)
            plt.plot(x, mean, label=f"({line_eq}) Bind: {cpu_omp_threads_bind}", marker='o', color=color)

        plt.plot([], [], color="grey", label='Fitted line', linestyle=':')
        plt.title(
            f"Batch Prefill Throughput vs. CPU OMP Thread Bind\n" \
            f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}\n" \
            f"Model: {model}, Number of Concurrent Reqs: {num_concurrent_requests}"
        )
        plt.xlabel('Number of Input Tokens')
        plt.ylabel('Throughput (tokens per second)')
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}/cpu_omp_threads_bind_prefill_throughput-{num_concurrent_requests}_reqs.png")
        plt.close()

def plot_cpu_omp_threads_binds_decode_throughput(output_dir: str, results: List[RequestData], metadata: dict):
    """
    Plot TTFT for varying input sizes.
    Graphs are separated based on the number of concurrent requests.
    """
    cpu_name, _, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)
    if not run_on_cpu:
        return

    output_dir = f"{output_dir}/cpu_omp_threads_bind_decode_throughput"
    os.makedirs(output_dir, exist_ok=True)

    min_input_tokens = min([r.num_input_tokens for r in results])
    for num_concurrent_requests in set([r.num_concurrent_requests for r in results]):
        color_cycle = get_colour_cycle()
        plt.figure(figsize=(10, 6))

        # Keep results with highest execution time
        filtered_results = list(filter(lambda r: r.num_input_tokens == 1, results))
        filtered_results = keep_per_request_batch(
            filtered_results, 
            lambda r: r.time_to_token_s[-1],
            keep_max=True
        )

        groups = defaultdict(list)
        for result in filtered_results:
            if result.num_concurrent_requests != num_concurrent_requests:
                continue
            groups[result.cpu_omp_threads_bind].append(result)
        for key in groups:
            groups[key].sort(key=lambda result: result.num_output_tokens)
    
        for cpu_omp_threads_bind, grouped_results in groups.items():
            x, y = [], []
            for result in grouped_results:
                x.append(result.num_output_tokens)
                throughput_tps = (result.num_concurrent_requests * result.num_output_tokens) / result.time_to_token_s[-1]
                y.append(throughput_tps)
            if not x:
                continue

            color = next(color_cycle)
            x, mean, std = format_multisample_data(x, y)
            plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
            line_eq = None # plot_fitted_line(color, x, mean)
            plt.plot(x, mean, label=f"({line_eq}) " if line_eq is not None else "" \
                     + f"Bind: {cpu_omp_threads_bind}", marker='o', color=color)

        plt.plot([], [], color="grey", label='Fitted line', linestyle=':')
        plt.title(
            f"Batch Decode Throughput vs. CPU OMP Thread Bind\n" \
            f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}\n" \
            f"Model: {model},  Input Tokens: {min_input_tokens},  Concurrent Reqs: {num_concurrent_requests}"
        )
        plt.xlabel('Number of Output Tokens')
        plt.ylabel('Throughput (tokens per second)')
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{output_dir}/cpu_omp_threads_bind_decode_throughput-{num_concurrent_requests}_reqs.png")
        plt.close()

def plot_prefill_throughput(output_dir: str, results: List[RequestData], metadata: dict, prefill_only: bool, per_request: bool):
    """
    For CPU-runs, the OMP thread bindings with the best top result is used.
    """
    cpu_name, gpu_name, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)

    # Filter by out results based on 'prefill_only' flag
    if prefill_only:
        results = list(filter(lambda r: r.num_output_tokens == 1, results))
    else:
        results = list(filter(lambda r: r.num_output_tokens > 1, results))

    # Define metric
    if per_request:
        metric_fn = lambda r: r.num_input_tokens / r.time_to_token_s[0]
    else:
        metric_fn = lambda r: (r.num_concurrent_requests * r.num_input_tokens / r.time_to_token_s[0])
        
        # Keep only the request in the batch with the max prefix time
        results = keep_per_request_batch(
            results, 
            lambda r: r.time_to_token_s[0],
            keep_max=True
        )
    
    # Group data
    groups, best_omp_thread_binds = group_and_find_best_records(
        data=results,
        group_by_fn=lambda r: r.num_input_tokens,
        sub_group_by_fn=lambda r: r.num_concurrent_requests,
        metric_fn=metric_fn,
        best_attr_fn=lambda r: r.cpu_omp_threads_bind,
        minimize=False,
    )

    # Plot the filtered groups
    plt.figure(figsize=(10, 6))
    color_cycle = get_colour_cycle()
    for num_input_tokens, group in groups.items():
        if not group:
            continue
        
        x, y = [], []
        for result in group:
            x.append(result.num_concurrent_requests)
            y.append(metric_fn(result))

        # Plot line
        x, mean, std = format_multisample_data(x, y)
        color = next(color_cycle)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
        line_eq = None # plot_fitted_line(color, x, mean)
        plt.plot(x, mean, label=f"({line_eq}) " if line_eq is not None else "" \
                 f"Input tokens: {num_input_tokens}", color=color, marker='o', ms=4)

        # Mark the results
        mark: Dict[int, str] = {}
        for result in group:
            mark[result.num_concurrent_requests] = MARKERS[best_omp_thread_binds.index(result.cpu_omp_threads_bind)]
        for i in range(len(x)):
            plt.scatter(x[i], mean[i], marker=mark[x[i]], color=color, s=50)

    plt.gca().add_artist(
        plt.legend(
            [plt.scatter([], [], marker=MARKERS[i], color="black") for i in range(len(best_omp_thread_binds))],
            [f"Thread Bind: {best_omp_thread_binds[i]}" for i in range(len(best_omp_thread_binds))],
            loc='upper right'
        )
    )

    plt.plot([], [], color="grey", label='Fitted line', linestyle=':')
    plt.title(
        f"{'Request' if per_request else 'Batch'} Prefill Throughput vs. Batch Size\n" \
        f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
        f"Model: {model}, Output Tokens: {1}"
    )
    plt.xlabel('Number of Requests (Batch Size)')
    plt.ylabel('Throughput (tokens per seconds)')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/prefill_throughput_{'request' if per_request else 'batch'}_{'prefill_only' if prefill_only else 'with_decode'}.png")
    plt.close()
    

def plot_decode_throughput(output_dir: str, results: List[RequestData], metadata: dict, per_request: bool):
    """
    For CPU-runs, the OMP thread bindings with the best top result is used.
    """
    cpu_name, gpu_name, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)

    min_input_tokens = min([r.num_input_tokens for r in results])
    results = list(filter(lambda r: r.num_input_tokens == min_input_tokens, results))
    assert len([r for r in results if r.num_input_tokens > min_input_tokens]) == 0

    # Metric
    if per_request:
        metric_fn = lambda r: (r.num_output_tokens / r.time_to_token_s[-1])
    else:
        metric_fn = lambda r: (r.num_concurrent_requests * r.num_output_tokens / r.time_to_token_s[-1])
        # Keep only the request in the batch with the max execution time
        results = keep_per_request_batch(
            results, 
            lambda r: r.time_to_token_s[-1],
            keep_max=True
        )

    # Group data
    groups, best_omp_thread_binds = group_and_find_best_records(
        data=results,
        group_by_fn=lambda r: r.num_output_tokens,
        sub_group_by_fn=lambda r: r.num_concurrent_requests,
        metric_fn=metric_fn,
        best_attr_fn=lambda r: r.cpu_omp_threads_bind,
        minimize=False,
    )

    # Plot the filtered groups
    plt.figure(figsize=(10, 6))
    mark: Dict[int, str] = {}
    color_cycle = get_colour_cycle()
    for num_output_tokens, group in groups.items():
        x, y = [], []
        for result in group:
            x.append(result.num_concurrent_requests)
            y.append(metric_fn(result))
        x, mean, std = format_multisample_data(x, y)
        
        color = next(color_cycle)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
        line_eq = None # plot_fitted_line(color, x, mean)
        plt.plot(x, mean, label=f"({line_eq}) " if line_eq is not None else "" \
                 f"Output tokens: {num_output_tokens}", color=color, marker='o', ms=4)

        # Mark the results
        for result in group:
            mark[result.num_concurrent_requests] = MARKERS[best_omp_thread_binds.index(result.cpu_omp_threads_bind)]
        for i in range(len(x)):
            if x[i] in mark:
                plt.scatter(x[i], mean[i], marker=mark[x[i]], color=color, s=70)
    
    plt.gca().add_artist(
        plt.legend(
            [plt.scatter([], [], marker=MARKERS[i], color="black") for i in range(len(best_omp_thread_binds))],
            [f"Thread Bind: {best_omp_thread_binds[i]}" for i in range(len(best_omp_thread_binds))],
            loc='upper right'
        )
    )

    plt.plot([], [], color="grey", label='Fitted line', linestyle=':')
    plt.title(
        f"{'Request' if per_request else 'Batch'} Decode Throughput vs. Batch Size\n" \
        f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
        f"Input Tokens: {min_input_tokens}, Model: {model}"
    )
    plt.xlabel('Number of Requests (Batch Size)')
    plt.ylabel('Throughput (tokens per second)')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/decode_throughput_{'request' if per_request else 'batch'}.png")
    plt.close()

def plot_tbot(output_dir: str, results: List[RequestData], metadata: dict):
    """
    Plot time between output tokens.
    For CPU-runs, the OMP thread bindings with the best top result is used.
    """
    cpu_name, gpu_name, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)

    min_input_tokens = min([r.num_input_tokens for r in results])
    results = list(filter(lambda r: r.num_input_tokens == min_input_tokens, results))

    max_output_tokens = max([r.num_output_tokens for r in results])
    results = list(filter(lambda r: r.num_output_tokens == max_output_tokens, results))

    # Group data
    groups, best_omp_thread_binds = group_and_find_best_records(
        data=results,
        group_by_fn=lambda r: r.num_concurrent_requests,
        sub_group_by_fn=lambda r: r.num_output_tokens,
        metric_fn=lambda r: r.time_to_token_s[-1],
        best_attr_fn=lambda r: r.cpu_omp_threads_bind,
        minimize=True,
    )

    # Plot the filtered groups
    plt.figure(figsize=(10, 6))
    mark: Dict[int, str] = {}
    color_cycle = get_colour_cycle()
    for num_reqs, group in groups.items():
        x, y = [], []
        for result in group:
            for i in range(1, result.num_output_tokens-1):
                if i >= len(result.time_to_token_s):
                    continue
                x.append(i+1)
                if (i == 0): # prefill
                    y.append(result.time_to_token_s[i])
                else: # decode
                    y.append(result.time_to_token_s[i] - result.time_to_token_s[i-1])
        x, mean, std = format_multisample_data(x, y)
        
        color = next(color_cycle)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
        line_eq = None # plot_fitted_line(color, x, mean)
        plt.plot(x, mean, label=f"({line_eq}) " if line_eq is not None else "" \
                 f"Num requests: {num_reqs}", marker='o', ms=4, color=color)
    
        # Mark the results
        for result in group:
            mark[result.num_output_tokens] = MARKERS[best_omp_thread_binds.index(result.cpu_omp_threads_bind)]
        for i in range(len(x)):
            if x[i] in mark:
                plt.scatter(x[i], mean[i], marker=mark[x[i]], color=color, s=70)
    
    plt.gca().add_artist(
        plt.legend(
            [plt.scatter([], [], marker=MARKERS[i], color="black") for i in range(len(best_omp_thread_binds))],
            [f"Thread Bind: {best_omp_thread_binds[i]}" for i in range(len(best_omp_thread_binds))],
            loc='upper right'
        )
    )

    plt.plot([], [], color="grey", label='Fitted line', linestyle=':')
    plt.title(
        f"Time Between Output Tokens vs. Output Token Length\n" \
        f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
        f"Input Tokens: {min_input_tokens}, Ouput Tokens: {max_output_tokens}, Model: {model}"
    )
    plt.xlabel('Number of Output Tokens')
    plt.ylabel('Time Since Last Token (seconds)')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/tbot.png")
    plt.close()

def plot_tbot_vs_num_requests(output_dir: str, results: List[RequestData], metadata: dict):
    """
    Plot change in time between output tokens.
    For CPU-runs, the OMP thread bindings with the best top result is used.
    """
    cpu_name, gpu_name, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)

    min_input_tokens = min([r.num_input_tokens for r in results])
    results = list(filter(lambda r: r.num_input_tokens == min_input_tokens, results))
    
    max_tbot_diff = 0
    best_omp_thread_binds = set()
    def get_data_for_output_tokens(output_tokens):
        nonlocal max_tbot_diff
        nonlocal best_omp_thread_binds
        filtered_results = list(filter(lambda r: r.num_output_tokens == output_tokens, results))

        groups, best_omp_thread_binds = group_and_find_best_records(
            data=filtered_results,
            group_by_fn=lambda r: r.num_concurrent_requests,
            sub_group_by_fn=lambda r: r.num_output_tokens,
            metric_fn=lambda r: r.time_to_token_s[-1],
            best_attr_fn=lambda r: r.cpu_omp_threads_bind,
            minimize=True,
        )

        # Gather points based on slope
        data_points_x = []
        data_points_y = []
        for num_reqs, group in groups.items():
            x, y = [], []
            for result in group:
                for i in range(1, result.num_output_tokens-1):
                    if i >= len(result.time_to_token_s):
                        continue
                    x.append(i+1)
                    if (i == 0): # prefill
                        y.append(result.time_to_token_s[i])
                    else: # decode
                        y.append(result.time_to_token_s[i] - result.time_to_token_s[i-1])
            
            if len(x) == 0:
                continue

            x, mean, _ = format_multisample_data(x, y)
            m, b = np.polyfit(x, mean, 1)
            max_tbot_diff = max(max_tbot_diff, m)
            data_points_x.append(num_reqs)
            data_points_y.append(b)
        return data_points_x, data_points_y

    # Plot
    plt.figure(figsize=(10, 6))
    for output_tokens in sorted(list(set(r.num_output_tokens for r in results))):
        x, y = get_data_for_output_tokens(output_tokens)
        assert len(x) == len(y)
        if len(x) > 0:
            plt.plot(x, y, marker='o', label=f"Output Tokens: {output_tokens}")
    plt.title(
        f"Time Between Output Token vs. Number of (Concurrent) Requests\n" \
        f"(TBOT change less than {max_tbot_diff:.5f} between consecutive output tokens)\n" \
        f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
        f"Input Tokens: {min_input_tokens}, Model: {model}" + f"{f', (best) OMP Thread Binds: {best_omp_thread_binds}' if run_on_cpu else ''}"
    )
    plt.subplots_adjust(top=0.85)
    plt.xlabel('Number of (Concurrent) Requests')
    plt.ylabel('Time Between Output Token (seconds)')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/tbot_vs_num_requests.png")
    plt.close()

def plot_histogram_of_samples_times(output_dir: str, results: List[RequestData], metadata: dict):
    """
    Plot a histogram of the samples, grouped by sample number.
    """
    unique_runs = sorted(list(set(r.run_num for r in results)))
    data_to_plot = [
        [r.time_to_token_s[-1] for r in results if r.run_num == s]
        for s in unique_runs
    ]

    plt.figure(figsize=(10, 6))
    plt.hist(
        data_to_plot,
        stacked=True,
        bins=50,
        label=[f"Run {s}" for s in unique_runs], # type: ignore
    )
    plt.title("Request Completion Times")
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/histogram_of_sample_times.png")
    plt.close()

def plot_energy_vs_num_reqs(output_dir: str, results: List[RequestData], metadata: dict, show_line_eq: bool, prefill: bool):
    """
    Plot energy per output/input token. (multiple lines for different request input/output lengths)
    """
    cpu_name, gpu_name, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)

    if prefill:
        results = list(filter(lambda r: r.num_output_tokens == 1, results))
    else:
        results = list(filter(lambda r: r.num_input_tokens == 1, results))
    
    if not results:
        return

    # Define metric to plot
    metric_fn = lambda r: float(r.request_batch_energy_joules)/float(
        r.num_concurrent_requests * (r.num_input_tokens if prefill else r.num_output_tokens)
    )

    # From request data from a batch, get the one with max value
    results = keep_per_request_batch(results, metric_fn, keep_max=True)

    # Group
    groups, best_omp_thread_binds = group_and_find_best_records(
        data=results,
        group_by_fn=lambda r: r.num_input_tokens if prefill else r.num_output_tokens,
        sub_group_by_fn=lambda r: r.num_concurrent_requests,
        metric_fn=metric_fn,
        best_attr_fn=lambda r: r.cpu_omp_threads_bind,
        minimize=True,
    )

    # Plot
    plt.figure(figsize=(10, 6))
    color_cycle = get_colour_cycle()
    for num_tokens, group in groups.items():
        x, y = [], []
        for result in group:
            x.append(result.num_concurrent_requests)
            y.append(metric_fn(result))
        if len(x) == 0:
            continue
        x, mean, std = format_multisample_data(x, y)
        color = next(color_cycle)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
        line_eq = None
        if show_line_eq:
            line_eq = plot_fitted_line(color, x, mean)
        plt.plot(
            x, mean,
            label=f"({line_eq}) " if line_eq is not None else "" + \
                f"Request num tokens: {num_tokens}",
            marker='o', color=color
        )

        # Mark the results
        mark = {}
        for result in group:
            mark[result.num_concurrent_requests] = MARKERS[best_omp_thread_binds.index(result.cpu_omp_threads_bind)]
        for i in range(len(x)):
            if x[i] in mark:
                plt.scatter(x[i], mean[i], marker=mark[x[i]], color=color, s=70)
    
    plt.gca().add_artist(
        plt.legend(
            [plt.scatter([], [], marker=MARKERS[i], color="black") for i in range(len(best_omp_thread_binds))],
            [f"Thread Bind: {best_omp_thread_binds[i]}" for i in range(len(best_omp_thread_binds))],
            loc='upper right'
        )
    )
    plt.title(
        f"Energy Per {'Input' if prefill else 'Output'} Token\n" \
        f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
        f"Model: {model}"
    )
    plt.xlabel('Number of Requests')
    plt.ylabel('Joules Per Token')
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/energy_per_{'input' if prefill else 'output'}_token.png")
    plt.close()

def plot_model_num_requests_vs_throughput(output_dir: str, results: List[RequestData], metadata: dict, prefill: bool):
    """
    Plot the throughput for prefill and decode with respect to the model. (summary graph)
    """
    cpu_name, gpu_name, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    
    # Filter
    if prefill:
        results = list(filter(lambda r: r.num_output_tokens == 1, results))
    else:
        results = list(filter(lambda r: r.num_input_tokens == 1, results))

    # From request data from a batch, get the one with max value
    results = keep_per_request_batch(
        results, 
        lambda r: r.time_to_token_s[0] if prefill else r.time_to_token_s[-1],
        keep_max=True
    )

    # Define metric
    metric_fn = lambda r: (
        r.num_concurrent_requests * (r.num_input_tokens if prefill else r.num_output_tokens)
        / (r.time_to_token_s[0] if prefill else r.time_to_token_s[-1])
    )

    # Gather max number of tokens to compare, then filter by the max
    model_to_tokens: Dict[str, set] = defaultdict(set)
    for r in results:
        model_to_tokens[r.model].add(r.num_input_tokens if prefill else r.num_output_tokens)

    max_token = max(set.intersection(*[l for _, l in model_to_tokens.items()]))
    if prefill:
        results = list(filter(lambda r: r.num_input_tokens == max_token, results))
    else:
        results = list(filter(lambda r: r.num_output_tokens == max_token, results))

    # Compute statistics
    groups, best_omp_thread_binds = group_and_find_best_records(
        data=results,
        group_by_fn=lambda r: r.model,
        sub_group_by_fn=lambda r: r.num_concurrent_requests,
        metric_fn=metric_fn,
        best_attr_fn=lambda r: r.cpu_omp_threads_bind,
        minimize=False,
    )

    # Plot
    plt.figure(figsize=(10, 6))
    plt.grid(True)

    mark = {}
    color_cycle = get_colour_cycle()
    for model, group in groups.items():
        x, y = [], []
        for result in group:
            x.append(result.num_concurrent_requests)
            y.append(metric_fn(result))
        x, mean, std = format_multisample_data(x, y)

        color = next(color_cycle)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
        line_eq = None # plot_fitted_line(color, x, mean)
        plt.plot(x, mean, label=f"({line_eq}) " if line_eq is not None else "" \
                 f"{model}", color=color, marker='o', ms=4)

        # Mark the results
        for result in group:
            mark[result.num_concurrent_requests] = MARKERS[best_omp_thread_binds.index(result.cpu_omp_threads_bind)]
        for i in range(len(x)):
            if x[i] in mark:
                plt.scatter(x[i], mean[i], marker=mark[x[i]], color=color, s=70)
    
    plt.gca().add_artist(
        plt.legend(
            [plt.scatter([], [], marker=MARKERS[i], color="black") for i in range(len(best_omp_thread_binds))],
            [f"Thread Bind: {best_omp_thread_binds[i]}" for i in range(len(best_omp_thread_binds))],
            loc='upper right'
        )
    )
    plt.title(
        f"Model vs. {'Prefill' if prefill else 'Decode'} Throughput\n" \
        f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
        f"Input Tokens: {max_token if prefill else 1}, Output Tokens: {1 if prefill else max_token}"
    )
    plt.xlabel('Number of Requests')
    plt.ylabel('Throughput (tokens per second)')
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(f"{output_dir}/model_vs_{'prefill' if prefill else 'decode'}_throughput.png")
    plt.close()

def plot_model_vs_throughput(output_dir: str, results: List[RequestData], metadata: dict, prefill: bool):
    """
    Plot the throughput for prefill and decode with respect to the model. (summary graph)
    """
    output_dir = f"{output_dir}/models"
    os.makedirs(output_dir, exist_ok=True)
    cpu_name, gpu_name, run_on_cpu, models = plot_utils.get_common_metadata(results, metadata)

    # Filter
    if prefill:
        results = list(filter(lambda r: r.num_output_tokens == 1, results))
    else:
        results = list(filter(lambda r: r.num_input_tokens == 1, results))

    # From request data from a batch, get the one with max value
    results = keep_per_request_batch(
        results,
        lambda r: r.time_to_token_s[0] if prefill else r.time_to_token_s[-1],
        keep_max=True
    )

    # Define metric
    metric_fn = lambda r: (
        r.num_concurrent_requests * (r.num_input_tokens if prefill else r.num_output_tokens) / r.time_to_token_s[-1]
    )

    # Gather max number of tokens to compare, then filter by the max
    model_to_tokens: Dict[str, set] = defaultdict(set)
    for r in results:
        model_to_tokens[r.model].add(r.num_input_tokens if prefill else r.num_output_tokens)

    max_token = max(set.intersection(*[l for _, l in model_to_tokens.items()]))
    if prefill:
        results = list(filter(lambda r: r.num_input_tokens == max_token, results))
    else:
        results = list(filter(lambda r: r.num_output_tokens == max_token, results))

    def plot_model_histogram(num_requests: int):
        nonlocal results
        filtered_results = list(filter(lambda r: r.num_concurrent_requests == num_requests, results))
        groups, best_omp_thread_binds = group_and_find_best_records(
            data=filtered_results,
            group_by_fn=lambda r: r.model,
            sub_group_by_fn=lambda r: r.num_concurrent_requests,
            metric_fn=metric_fn,
            best_attr_fn=lambda r: r.cpu_omp_threads_bind,
            minimize=False,
        )
        
        plt.figure(figsize=(10, 6))
        x, y = [], []
        for model, group_results in groups.items():
            assert all(group_results[0].num_concurrent_requests == r.num_concurrent_requests for r in group_results)
            for r in group_results:
                assert r.model == model
                x.append(model) # turn model names into indices for grouping
                y.append(metric_fn(r))
        if not x:
            return
        x, mean, std = format_multisample_data(x, y)
        x, mean, std = zip(*sorted(zip(x, mean, std), key=lambda z: models.index(z[0]))) # Sort to maintain model order
        plt.bar(list(x), list(mean), yerr=std)
        plt.title(
            f"Model vs. {'Prefill' if prefill else 'Decode'} Throughput ({num_requests} requests)\n" \
            f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
            f"Input Tokens: {max_token if prefill else 1}, Output Tokens: {1 if prefill else max_token}" \
            f"{f', (best) OMP Thread Binds: {best_omp_thread_binds}' if run_on_cpu else ''}"
        )
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('Throughput (tokens per second)')
        plt.xlabel('Model Name')
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.savefig(f"{output_dir}/model_vs_{'prefill' if prefill else 'decode'}_throughput-{num_requests}_requests.png", bbox_inches='tight')
        plt.close()

    all_num_requests = set([r.num_concurrent_requests for r in results])
    for num_requests in all_num_requests:
        plot_model_histogram(num_requests)


def plot_hardware_energy(output_dir: str, results: List[RequestData], emissions: Dict[str, EmissionsData], metadata: dict, prefill: bool):
    """
    Plot the utility of energy components 
    """
    cpu_name, gpu_name, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)

    output_dir = f"{output_dir}/hardware_energy"
    os.makedirs(output_dir, exist_ok=True)

    # Filter to keep only the largest prefill and decode requests
    if prefill:
        results = list(filter(lambda r: r.num_output_tokens == 1, results))
    else:
        results = list(filter(lambda r: r.num_input_tokens == 1, results))

    # Plot for each token size
    tokens_list = set([r.num_input_tokens for r in results] if prefill else [r.num_output_tokens for r in results])
    for token_num in tokens_list:
        filtered_results = list(filter(
            lambda r: r.num_input_tokens == token_num if prefill else r.num_output_tokens == token_num,
            results
        ))

        # Keep only one request per batch for reference
        filtered_results = keep_per_request_batch(filtered_results, lambda _: 1, keep_max=True)

        # Group data
        groups, best_omp_thread_binds = group_and_find_best_records(
            data=filtered_results,
            group_by_fn=lambda r: r.num_input_tokens if prefill else r.num_output_tokens,
            sub_group_by_fn=lambda r: r.num_concurrent_requests,
            metric_fn=lambda r: emissions[r.request_batch_uid].cpu_energy + emissions[r.request_batch_uid].gpu_energy,
            best_attr_fn=lambda r: r.cpu_omp_threads_bind,
            minimize=True,
        )

        # Compile lines to draw
        plt.figure(figsize=(10, 6))
        color_cycle = get_colour_cycle()
        for num_tokens, group in groups.items():
            if not group:
                continue
            x_cpu, x_gpu, y_cpu, y_gpu = [], [], [], []
            for result in group:
                emissions_obj = emissions[result.request_batch_uid]
                num_batch_tokens = result.num_concurrent_requests * (result.num_input_tokens if prefill else result.num_output_tokens)
                x_cpu.append(result.num_concurrent_requests)
                y_cpu.append(emissions_obj.cpu_energy * 3600 * 1000 / num_batch_tokens)
                x_gpu.append(result.num_concurrent_requests)
                y_gpu.append(emissions_obj.gpu_energy * 3600 * 1000 / num_batch_tokens)

            x_cpu, mean_cpu, std_cpu = format_multisample_data(x_cpu, y_cpu)
            x_gpu, mean_gpu, std_gpu = format_multisample_data(x_gpu, y_gpu)

            color = next(color_cycle)
            plt.plot(x_cpu, mean_cpu, linestyle='dashed', color=color, marker="o")
            plt.fill_between(x_cpu, mean_cpu - std_cpu, mean_cpu + std_cpu, alpha=0.2, color=color)
            plt.plot(x_gpu, mean_gpu, linestyle='dotted', color=color, marker="^")
            plt.fill_between(x_gpu, mean_gpu - std_gpu, mean_gpu + std_gpu, alpha=0.2, color=color)

        # Plot
        plt.plot([], [], color="grey", label='CPU', linestyle='dashed', marker="o")
        plt.plot([], [], color="grey", label='GPU', linestyle='dotted', marker="^")
        plt.title(
            f"Joules Per {'Prefill' if prefill else 'Decode'} Token Per Hardware\n" \
            f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
            f"Input Tokens: {token_num if prefill else 1}, Output Tokens: {1 if prefill else token_num}" \
            f"{f', (best) OMP Thread Binds: {best_omp_thread_binds}' if run_on_cpu else ''}"
        )
        plt.ylabel('Energy Per Token (joules)')
        plt.xlabel('Number of (Concurrent) Requests')
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.legend()

        plt.savefig(f"{output_dir}/hardware_joules_per_{'prefill' if prefill else 'decode'}_token-{token_num}_tokens.png", bbox_inches='tight')
        plt.close()

def plot_hardware_wattage(output_dir: str, results: List[RequestData], emissions: Dict[str, EmissionsData], metadata: dict, prefill: bool):
    """
    Plot the utility of energy components 
    """
    cpu_name, gpu_name, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)

    output_dir = f"{output_dir}/hardware_wattage"
    os.makedirs(output_dir, exist_ok=True)

    # Filter to keep only the largest prefill and decode requests
    if prefill:
        results = list(filter(lambda r: r.num_output_tokens == 1, results))
    else:
        results = list(filter(lambda r: r.num_input_tokens == 1, results))

    # Plot for each token size
    tokens_list = set([r.num_input_tokens for r in results] if prefill else [r.num_output_tokens for r in results])
    for token_num in tokens_list:
        filtered_results = list(filter(
            lambda r: r.num_input_tokens == token_num if prefill else r.num_output_tokens == token_num,
            results
        ))

        # Keep only one request per batch for reference
        filtered_results = keep_per_request_batch(filtered_results, lambda _: 1, keep_max=True)

        # Group data
        groups, best_omp_thread_binds = group_and_find_best_records(
            data=filtered_results,
            group_by_fn=lambda r: r.num_input_tokens if prefill else r.num_output_tokens,
            sub_group_by_fn=lambda r: r.num_concurrent_requests,
            metric_fn=lambda r: \
                emissions[r.request_batch_uid].cpu_power + \
                emissions[r.request_batch_uid].gpu_power + \
                emissions[r.request_batch_uid].ram_power,
            best_attr_fn=lambda r: r.cpu_omp_threads_bind,
            minimize=True,
        )

        # Compile lines to draw
        plt.figure(figsize=(10, 6))
        color_cycle = get_colour_cycle()
        for num_tokens, group in groups.items():
            if not group:
                continue
            x_cpu, x_gpu, x_ram, y_cpu, y_gpu, y_ram = [], [], [], [], [], []
            for result in group:
                emissions_obj = emissions[result.request_batch_uid]
                num_batch_tokens = result.num_concurrent_requests * (result.num_input_tokens if prefill else result.num_output_tokens)
                x_cpu.append(result.num_concurrent_requests)
                y_cpu.append(emissions_obj.cpu_power)
                x_gpu.append(result.num_concurrent_requests)
                y_gpu.append(emissions_obj.gpu_power)
                x_ram.append(result.num_concurrent_requests)
                y_ram.append(emissions_obj.ram_power)

            x_cpu, mean_cpu, std_cpu = format_multisample_data(x_cpu, y_cpu)
            x_gpu, mean_gpu, std_gpu = format_multisample_data(x_gpu, y_gpu)
            x_ram, mean_ram, std_ram = format_multisample_data(x_ram, y_ram)

            color = next(color_cycle)
            plt.plot(x_cpu, mean_cpu, linestyle='dashed', color=color, marker="o")
            plt.fill_between(x_cpu, mean_cpu - std_cpu, mean_cpu + std_cpu, alpha=0.2, color=color)
            plt.plot(x_gpu, mean_gpu, linestyle='dotted', color=color, marker="^")
            plt.fill_between(x_gpu, mean_gpu - std_gpu, mean_gpu + std_gpu, alpha=0.2, color=color)
            plt.plot(x_ram, mean_ram, linestyle='solid', color=color, marker="*")
            plt.fill_between(x_ram, mean_ram - std_ram, mean_ram + std_ram, alpha=0.2, color=color)

        # Plot
        plt.plot([], [], color="grey", label='CPU', linestyle='dashed', marker="o")
        plt.plot([], [], color="grey", label='GPU', linestyle='dotted', marker="^")
        plt.plot([], [], color="grey", label='RAM', linestyle='solid', marker="*")
        plt.title(
            f"Average Wattage for {'Prefill' if prefill else 'Decode'} Requests\n" \
            f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
            f"Input Tokens: {token_num if prefill else 1}, Output Tokens: {1 if prefill else token_num}" \
            f"{f', (best) OMP Thread Binds: {best_omp_thread_binds}' if run_on_cpu else ''}"
        )
        plt.ylabel('Average Wattage (watts)')
        plt.xlabel('Number of (Concurrent) Requests')
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.legend()

        plt.savefig(f"{output_dir}/hardware_wattage_{'prefill' if prefill else 'decode'}_token-{token_num}_tokens.png", bbox_inches='tight')
        plt.close()

def plot_hardware_utility(output_dir: str, results: List[RequestData], emissions: Dict[str, EmissionsData], metadata: dict, prefill: bool):
    """
    Plot the utility of hardware components 
    """
    cpu_name, gpu_name, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)

    output_dir = f"{output_dir}/hardware_utility"
    os.makedirs(output_dir, exist_ok=True)

    # Filter to keep only the largest prefill and decode requests
    if prefill:
        results = list(filter(lambda r: r.num_output_tokens == 1, results))
    else:
        results = list(filter(lambda r: r.num_input_tokens == 1, results))

    # Plot for each token size
    tokens_list = set([r.num_input_tokens for r in results] if prefill else [r.num_output_tokens for r in results])
    for token_num in tokens_list:
        filtered_results = list(filter(
            lambda r: r.num_input_tokens == token_num if prefill else r.num_output_tokens == token_num,
            results
        ))

        # Keep only one request per batch for reference
        filtered_results = keep_per_request_batch(filtered_results, lambda _: 1, keep_max=True)

        # Group data
        groups, best_omp_thread_binds = group_and_find_best_records(
            data=filtered_results,
            group_by_fn=lambda r: r.num_input_tokens if prefill else r.num_output_tokens,
            sub_group_by_fn=lambda r: r.num_concurrent_requests,
            metric_fn=lambda r: emissions[r.request_batch_uid].cpu_utilization_percent,
            best_attr_fn=lambda r: r.cpu_omp_threads_bind,
            minimize=True,
        )

        # Compile lines to draw
        plt.figure(figsize=(10, 6))
        color_cycle = get_colour_cycle()
        for num_tokens, group in groups.items():
            if not group:
                continue
            x_cpu, x_gpu, y_cpu, y_gpu = [], [], [], []
            for result in group:
                emissions_obj = emissions[result.request_batch_uid]
                x_cpu.append(result.num_concurrent_requests)
                y_cpu.append(emissions_obj.cpu_utilization_percent)
                x_gpu.append(result.num_concurrent_requests)
                y_gpu.append(emissions_obj.gpu_utilization_percent)

            x_cpu, mean_cpu, std_cpu = format_multisample_data(x_cpu, y_cpu)
            x_gpu, mean_gpu, std_gpu = format_multisample_data(x_gpu, y_gpu)

            color = next(color_cycle)
            plt.plot(x_cpu, mean_cpu, linestyle='dashed', color=color, marker="o")
            plt.fill_between(x_cpu, mean_cpu - std_cpu, mean_cpu + std_cpu, alpha=0.2, color=color)
            plt.plot(x_gpu, mean_gpu, linestyle='dotted', color=color, marker="^")
            plt.fill_between(x_gpu, mean_gpu - std_gpu, mean_gpu + std_gpu, alpha=0.2, color=color)

        # Plot
        plt.plot([], [], color="grey", label='CPU', linestyle='dashed', marker="o")
        plt.plot([], [], color="grey", label='GPU', linestyle='dotted', marker="^")
        plt.title(
            f"Percentage Utilization for {'Prefill' if prefill else 'Decode'} Requests Per Hardware\n" \
            f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
            f"Input Tokens: {token_num if prefill else 1}, Output Tokens: {1 if prefill else token_num}" \
            f"{f', (best) OMP Thread Binds: {best_omp_thread_binds}' if run_on_cpu else ''}"
        )
        plt.ylabel('Percentage Utilization')
        plt.xlabel('Number of (Concurrent) Requests')
        plt.ylim(bottom=0)
        plt.grid(True)
        plt.legend()

        plt.savefig(f"{output_dir}/hardware_utility-{'prefill' if prefill else 'decode'}-{token_num}_tokens.png", bbox_inches='tight')
        plt.close()



if __name__ == "__main__":
    # Load data
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results-path', type=str, default=None, help='Path to a specific results directory.')
    args = parser.parse_args()

    print(f"Plotting results: '{args.results_path}'")
    try:
        results = plot_utils.load_csv_data(args.results_path)
        metadata = metadata_util.load_metadata(args.results_path)
        emissions = plot_utils.load_csv_emissions(args.results_path)
    except FileNotFoundError:
        print(f"File not found.")
        exit(1)
    
    # General plots (involving all models)
    plots_dir = f"{args.results_path}"
    os.makedirs(plots_dir, exist_ok=True)
    plot_model_num_requests_vs_throughput(plots_dir, results, metadata, prefill=False)
    plot_model_num_requests_vs_throughput(plots_dir, results, metadata, prefill=True)
    plot_model_vs_throughput(plots_dir, results, metadata, prefill=False)
    plot_model_vs_throughput(plots_dir, results, metadata, prefill=True)

    # Group by model name
    model_groups = defaultdict(list)
    for result in results:
        model_groups[result.model].append(result)

    for model, model_results in model_groups.items():
        model_plots_dir = f"{plots_dir}/{model}"
        os.makedirs(model_plots_dir, exist_ok=True)
        plot_cpu_omp_threads_binds_prefill_throughput(model_plots_dir, model_results, metadata)
        plot_cpu_omp_threads_binds_decode_throughput(model_plots_dir, model_results, metadata)
        plot_prefill_throughput(model_plots_dir, model_results, metadata, prefill_only=True, per_request=False)
        plot_prefill_throughput(model_plots_dir, model_results, metadata, prefill_only=False, per_request=False)
        plot_prefill_throughput(model_plots_dir, model_results, metadata, prefill_only=True, per_request=True)
        plot_prefill_throughput(model_plots_dir, model_results, metadata, prefill_only=False, per_request=True)
        plot_decode_throughput(model_plots_dir, model_results, metadata, per_request=True)
        plot_decode_throughput(model_plots_dir, model_results, metadata, per_request=False)
        plot_tbot(model_plots_dir, model_results, metadata)
        plot_tbot_vs_num_requests(model_plots_dir, model_results, metadata)
        plot_histogram_of_samples_times(model_plots_dir, model_results, metadata)
        plot_energy_vs_num_reqs(model_plots_dir, model_results, metadata, show_line_eq=False, prefill=False)
        plot_energy_vs_num_reqs(model_plots_dir, model_results, metadata, show_line_eq=False, prefill=True)
        if emissions:
            plot_hardware_wattage(model_plots_dir, model_results, emissions, metadata, prefill=False)
            plot_hardware_wattage(model_plots_dir, model_results, emissions, metadata, prefill=True)
            plot_hardware_energy(model_plots_dir, model_results, emissions, metadata, prefill=False)
            plot_hardware_energy(model_plots_dir, model_results, emissions, metadata, prefill=True)
            plot_hardware_utility(model_plots_dir, model_results, emissions, metadata, prefill=False)
            plot_hardware_utility(model_plots_dir, model_results, emissions, metadata, prefill=True)

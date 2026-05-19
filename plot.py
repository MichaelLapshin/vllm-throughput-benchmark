import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import defaultdict
import argparse
import os

from results import RequestData
import pandas as pd
import numpy as np

from run_constants import RESULTS_DIR, PLOTS_DIR
from utils import plot_utils
from utils.plot_utils import MARKERS, group_and_find_best_records


def plot_fitted_line(color, x, y) -> str:
    """
    Plot fitted line, return formula
    """
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, [m * xv + b for xv in x], color=color, linestyle=':')
    return f"{m:.4f}n + {b:.4f}"

def get_poly_colour_no_alpha(poly):
    r, g, b, _ = poly.get_facecolor()[0]
    color_no_alpha = (r, g, b)
    return color_no_alpha

def format_multisample_data(x: list, y: list) -> Tuple:
    df = pd.DataFrame({'x': x, 'y': y})
    stats = df.groupby('x')['y'].agg(['mean', 'std']).reset_index()
    return (
        np.array(stats['x'].tolist()),
        np.array(stats['mean'].tolist()),
        np.array(stats['std'].tolist())
    )

def plot_cpu_omp_threads_binds_ttft(output_dir: str, results: List[RequestData], metadata: dict):
    """
    Plot TTFT for varying input sizes.
    Graphs are separated based on the number of concurrent requests.
    """
    cpu_name, _, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)
    if not run_on_cpu:
        return

    output_dir = f"{output_dir}/cpu_omp_threads_bind_ttft"
    os.makedirs(output_dir, exist_ok=True)

    for num_concurrent_requests in set([r.num_concurrent_requests for r in results]):
        plt.figure(figsize=(10, 6))

        groups = defaultdict(list)
        for result in results:
            if result.num_concurrent_requests != num_concurrent_requests:
                continue
            groups[result.cpu_omp_threads_bind].append(result)
        for key in groups:
            groups[key].sort(key=lambda result: result.num_input_tokens)
    
        for cpu_omp_threads_bind, grouped_results in groups.items():
            x, y = [], []
            for result in grouped_results:
                x.append(result.num_input_tokens)
                y.append(result.time_to_token_s[0])
            x, mean, std = format_multisample_data(x, y)
            
            poly = plt.fill_between(x, mean - std, mean + std, alpha=0.2)
            color = get_poly_colour_no_alpha(poly)
            line_eq = plot_fitted_line(color, x, mean)
            plt.plot(x, mean, label=f"({line_eq}) Bind: {cpu_omp_threads_bind}", marker='o', color=color)

        plt.plot([], [], color="grey", label='Fitted line', linestyle=':')
        plt.title(
            f"Request TTFT vs. CPU OMP Thread Bind\n" \
            f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}\n" \
            f"Model: {model}, Number of Concurrent Reqs: {num_concurrent_requests}"
        )
        plt.xlabel('Number of Input Tokens')
        plt.ylabel('Time to First Token (seconds)')
        plt.ylim(bottom=0)
        plt.legend()
        plt.savefig(f"{output_dir}/cpu_omp_threads_bind_ttft-{num_concurrent_requests}_reqs.png")
        plt.close()

def plot_cpu_omp_threads_binds_ttlt(output_dir: str, results: List[RequestData], metadata: dict):
    """
    Plot TTFT for varying input sizes.
    Graphs are separated based on the number of concurrent requests.
    """
    cpu_name, _, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)
    if not run_on_cpu:
        return

    output_dir = f"{output_dir}/cpu_omp_threads_bind_ttlt"
    os.makedirs(output_dir, exist_ok=True)

    min_input_tokens = min([r.num_input_tokens for r in results])
    for num_concurrent_requests in set([r.num_concurrent_requests for r in results]):
        plt.figure(figsize=(10, 6))

        groups = defaultdict(list)
        for result in results:
            if result.num_concurrent_requests != num_concurrent_requests:
                continue
            if result.num_input_tokens != min_input_tokens:
                continue
            groups[result.cpu_omp_threads_bind].append(result)
        for key in groups:
            groups[key].sort(key=lambda result: result.num_output_tokens)
    
        for cpu_omp_threads_bind, grouped_results in groups.items():
            x, y = [], []
            for result in grouped_results:
                for i in range(result.num_output_tokens):
                    if i < len(result.time_to_token_s):
                        x.append(i+1)
                        y.append(result.time_to_token_s[i])
            if not x:
                continue

            x, mean, std = format_multisample_data(x, y)
            poly = plt.fill_between(x, mean - std, mean + std, alpha=0.2)
            color = get_poly_colour_no_alpha(poly)
            line_eq = plot_fitted_line(color, x, mean)
            plt.plot(x, mean, label=f"({line_eq}) Bind: {cpu_omp_threads_bind}", marker='o', color=color)

        plt.plot([], [], color="grey", label='Fitted line', linestyle=':')
        plt.title(
            f"Request Time to Token vs. CPU OMP Thread Bind\n" \
            f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}\n" \
            f"Model: {model},  Input Tokens: {min_input_tokens},  Concurrent Reqs: {num_concurrent_requests}"
        )
        plt.xlabel('Number of Output Tokens')
        plt.ylabel('Time (seconds)')
        plt.ylim(bottom=0)
        plt.legend()
        plt.savefig(f"{output_dir}/cpu_omp_threads_bind_ttlt-{num_concurrent_requests}_reqs.png")
        plt.close()

def plot_ttft(output_dir: str, results: List[RequestData], metadata: dict, prefill_only: bool):
    """
    Plot time to first token.
    For CPU-runs, the OMP thread bindings with the best top result is used.
    """
    cpu_name, gpu_name, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)

    # Filter by out results based on 'prefill_only' flag
    if prefill_only:
        results = list(filter(lambda r: r.num_output_tokens == 1, results))
    else:
        results = list(filter(lambda r: r.num_output_tokens > 1, results))

    # Group data
    groups, best_omp_thread_binds = group_and_find_best_records(
        data=results,
        group_by_fn=lambda r: r.num_concurrent_requests,
        sub_group_by_fn=lambda r: r.num_input_tokens,
        metric_fn=lambda r: r.time_to_token_s[0],
        best_attr_fn=lambda r: r.cpu_omp_threads_bind,
    )

    # Plot the filtered groups
    plt.figure(figsize=(10, 6))
    for num_reqs, group in groups.items():
        if not group:
            continue
        
        x, y = [], []
        for result in group:
            x.append(result.num_input_tokens)
            y.append(result.time_to_token_s[0])

        # Plot line
        x, mean, std = format_multisample_data(x, y)
        poly = plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        color = get_poly_colour_no_alpha(poly)
        line_eq = plot_fitted_line(color, x, mean)
        plt.plot(x, mean, label=f"({line_eq}) Num requests: {num_reqs}", color=color, marker='o', ms=4)

        # Mark the results
        mark: Dict[int, str] = {}
        for result in group:
            mark[result.num_input_tokens] = MARKERS[best_omp_thread_binds.index(result.cpu_omp_threads_bind)]
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
        f"Request TTFT vs. Input Token Length\n" \
        f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
        f"Model: {model}"
    )
    plt.xlabel('Number of Input Tokens')
    plt.ylabel('Time (seconds)')
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(f"{output_dir}/ttft{'_prefill_only' if prefill_only else '_with_decode'}.png")
    plt.close()
    

def plot_ttlt(output_dir: str, results: List[RequestData], metadata: dict):
    """
    Plot time to last token.
    For CPU-runs, the OMP thread bindings with the best top result is used.
    """
    cpu_name, gpu_name, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    model = plot_utils.get_model(results)

    min_input_tokens = min([r.num_input_tokens for r in results])
    results = list(filter(lambda r: r.num_input_tokens == min_input_tokens, results))
    assert len([r for r in results if r.num_input_tokens > min_input_tokens]) == 0

    # Group data
    groups, best_omp_thread_binds = group_and_find_best_records(
        data=results,
        group_by_fn=lambda r: r.num_concurrent_requests,
        sub_group_by_fn=lambda r: r.num_output_tokens,
        metric_fn=lambda r: r.time_to_token_s[-1],
        best_attr_fn=lambda r: r.cpu_omp_threads_bind,
    )

    # Plot the filtered groups
    plt.figure(figsize=(10, 6))
    mark: Dict[int, str] = {}
    for num_reqs, group in groups.items():
        x, y = [], []
        for result in group:
            # x.append(result.num_output_tokens)
            # y.append(result.time_to_token_s[-1])
            for i in range(result.num_output_tokens):
                if i < len(result.time_to_token_s):
                    x.append(i+1)
                    y.append(result.time_to_token_s[i])
        x, mean, std = format_multisample_data(x, y)
        
        poly = plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        color = get_poly_colour_no_alpha(poly)
        line_eq = plot_fitted_line(color, x, mean)
        plt.plot(x, mean, label=f"({line_eq}) Num requests: {num_reqs}", color=color, marker='o', ms=4)

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
        f"Request Time to Token vs. Output Token Length\n" \
        f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
        f"Input Tokens: {min_input_tokens}, Model: {model}"
    )
    plt.xlabel('Number of Output Tokens')
    plt.ylabel('Time (seconds)')
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(f"{output_dir}/ttlt.png")
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
    )

    # Plot the filtered groups
    plt.figure(figsize=(10, 6))
    mark: Dict[int, str] = {}
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
        
        poly = plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        color = get_poly_colour_no_alpha(poly)
        line_eq = plot_fitted_line(color, x, mean)
        plt.plot(x, mean, label=f"({line_eq}) Num requests: {num_reqs}", marker='o', ms=4, color=color)
    
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

    # Group
    metric_func = lambda r: float(r.request_batch_energy_joules)/float(
        r.num_concurrent_requests * (r.num_input_tokens if prefill else r.num_output_tokens)
    )
    groups, best_omp_thread_binds = group_and_find_best_records(
        data=results,
        group_by_fn=lambda r: r.num_input_tokens if prefill else r.num_output_tokens,
        sub_group_by_fn=lambda r: r.num_concurrent_requests,
        metric_fn=metric_func,
        best_attr_fn=lambda r: r.cpu_omp_threads_bind,
    )

    # Plot
    plt.figure(figsize=(10, 6))
    for num_tokens, group in groups.items():
        x, y = [], []
        for result in group:
            x.append(result.num_concurrent_requests)
            y.append(metric_func(result))
        if len(x) == 0:
            continue
        x, mean, std = format_multisample_data(x, y)
        poly = plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        color = get_poly_colour_no_alpha(poly)
        line_eq = None
        if show_line_eq:
            line_eq = plot_fitted_line(color, x, mean)
        plt.plot(
            x, mean,
            label=f"({line_eq}) " if line_eq is not None else "" + f"Request num tokens: {num_tokens}",
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
    plt.legend()
    plt.savefig(f"{output_dir}/energy_per_{'input' if prefill else 'output'}_token.png")
    plt.close()

def plot_model_num_requests_vs_time(output_dir: str, results: List[RequestData], metadata: dict, prefill: bool):
    """
    Plot the execution time for prefill and decode with respect to the model. (summary graph)
    """
    cpu_name, gpu_name, run_on_cpu, _ = plot_utils.get_common_metadata(results, metadata)
    
    # Filter
    if prefill:
        results = list(filter(lambda r: r.num_output_tokens == 1, results))
    else:
        results = list(filter(lambda r: r.num_input_tokens == 1, results))

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
        metric_fn=lambda r: r.time_to_token_s[-1],
        best_attr_fn=lambda r: r.cpu_omp_threads_bind,
    )

    # Plot
    plt.figure(figsize=(10, 6))
    mark = {}
    for model, group in groups.items():
        x, y = [], []
        for result in group:
            x.append(result.num_concurrent_requests)
            y.append(result.time_to_token_s[-1])
        x, mean, std = format_multisample_data(x, y)

        poly = plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        color = get_poly_colour_no_alpha(poly)
        line_eq = plot_fitted_line(color, x, mean)
        plt.plot(x, mean, label=f"({line_eq}) {model}", color=color, marker='o', ms=4)

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
    plt.title(
        f"Model vs. {'Prefill' if prefill else 'Decode'} Time\n" \
        f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
        f"Input Tokens: {max_token if prefill else 1}, Output Tokens: {1 if prefill else max_token}"
    )
    plt.xlabel('Number of Requests')
    plt.ylabel('Time (second)')
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(f"{output_dir}/model_vs_{'prefill' if prefill else 'decode'}_time.png")
    plt.close()

def plot_model_size_vs_time(output_dir: str, results: List[RequestData], metadata: dict, prefill: bool):
    """
    Plot the execution time for prefill and decode with respect to the model. (summary graph)
    """
    output_dir = f"{output_dir}/models"
    os.makedirs(output_dir, exist_ok=True)
    cpu_name, gpu_name, run_on_cpu, models = plot_utils.get_common_metadata(results, metadata)

    # Filter
    if prefill:
        results = list(filter(lambda r: r.num_output_tokens == 1, results))
    else:
        results = list(filter(lambda r: r.num_input_tokens == 1, results))

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
            metric_fn=lambda r: r.time_to_token_s[-1],
            best_attr_fn=lambda r: r.cpu_omp_threads_bind,
        )
        
        plt.figure(figsize=(10, 6))
        x, y = [], []
        for model, group_results in groups.items():
            assert all(group_results[0].num_concurrent_requests == r.num_concurrent_requests for r in group_results)
            for r in group_results:
                assert r.model == model
                x.append(model) # turn model names into indices for grouping
                y.append(r.time_to_token_s[-1])
        if not x:
            return
        x, mean, std = format_multisample_data(x, y)
        x, mean, std = zip(*sorted(zip(x, mean, std), key=lambda z: models.index(z[0]))) # Sort to maintain model order
        plt.bar(list(x), list(mean), yerr=std)
        plt.title(
            f"Model size vs. {'Prefill' if prefill else 'Decode'} Time ({num_requests} requests)\n" \
            f"CPU using {metadata['environment']['TORCH_CPU_AVX']}: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ''}" + "\n" \
            f"Input Tokens: {max_token if prefill else 1}, Output Tokens: {1 if prefill else max_token}" \
            f"{f', (best) OMP Thread Binds: {best_omp_thread_binds}' if run_on_cpu else ''}"
        )
        plt.xticks(rotation=30, ha='right')
        plt.ylabel('Time (second)')
        plt.xlabel('Model Name')
        plt.ylim(bottom=0)
        plt.savefig(f"{output_dir}/model_vs_{'prefill' if prefill else 'decode'}_time-{num_requests}_requests.png", bbox_inches='tight')
        plt.close()

    all_num_requests = set([r.num_concurrent_requests for r in results])
    for num_requests in all_num_requests:
        plot_model_histogram(num_requests)


if __name__ == "__main__":
    # Load data
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default=None, help='Results directory name.')
    parser.add_argument('-a', '--all', action='store_true', help='Plot for all result directories.')
    args = parser.parse_args()

    if args.name is None:
        # Get latest directory in RESULTS_DIR
        results_dirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
        args.name = max(results_dirs, key=lambda d: os.path.getctime(os.path.join(RESULTS_DIR, d)))

    if args.all:
        results_dir_names = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    else:
        results_dir_names = [args.name]

    for results_name in results_dir_names:
        print(f"Plotting results: '{results_name}'")
        try:
            results = plot_utils.load_csv_data(results_name)
            metadata = plot_utils.load_metadata(results_name)
        except FileNotFoundError:
            print(f"File not found. Skipping...")
            continue
        
        # General plots (involving all models)
        output_dir = f"{PLOTS_DIR}/{results_name}"
        os.makedirs(output_dir, exist_ok=True)
        plot_model_num_requests_vs_time(output_dir, results, metadata, prefill=False)
        plot_model_num_requests_vs_time(output_dir, results, metadata, prefill=True)
        plot_model_size_vs_time(output_dir, results, metadata, prefill=False)
        plot_model_size_vs_time(output_dir, results, metadata, prefill=True)

        # Group by model name
        model_groups = defaultdict(list)
        for result in results:
            model_groups[result.model].append(result)

        for model, model_results in model_groups.items():
            model_output_dir = f"{output_dir}/{model}"
            os.makedirs(model_output_dir, exist_ok=True)
            plot_cpu_omp_threads_binds_ttft(model_output_dir, model_results, metadata)
            plot_cpu_omp_threads_binds_ttlt(model_output_dir, model_results, metadata)
            plot_ttft(model_output_dir, model_results, metadata, prefill_only=True)
            plot_ttft(model_output_dir, model_results, metadata, prefill_only=False)
            plot_ttlt(model_output_dir, model_results, metadata)
            plot_tbot(model_output_dir, model_results, metadata)
            plot_tbot_vs_num_requests(model_output_dir, model_results, metadata)
            plot_histogram_of_samples_times(model_output_dir, model_results, metadata)
            plot_energy_vs_num_reqs(model_output_dir, model_results, metadata, show_line_eq=False, prefill=False)
            plot_energy_vs_num_reqs(model_output_dir, model_results, metadata, show_line_eq=False, prefill=True)

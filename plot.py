import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List, Tuple, Dict
from dacite import from_dict
from collections import defaultdict
import argparse
import os
from ast import literal_eval

from run_constants import RESULTS_DIR, PLOTS_DIR
from results import Result
import pandas as pd
import numpy as np

def load_csv_data(results_name: str) -> List[Result]:
    results: List[Result] = []
    with open(f"{RESULTS_DIR}/{results_name}/data.csv", newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            for key in row:
                if Result.__dataclass_fields__[key].type == bool:
                    row[key] = bool(row[key])
                elif Result.__dataclass_fields__[key].type == int:
                    row[key] = int(row[key])
                elif Result.__dataclass_fields__[key].type == float:
                    row[key] = float(row[key])
                elif Result.__dataclass_fields__[key].type == List[float]:
                    row[key] = literal_eval(row[key])
            results.append(from_dict(data_class=Result, data=row))
    return results

def get_common_parameters(results: List[Result]):
    assert len(results) > 0
    model = results[0].model
    cpu_name = results[0].cpu_name
    gpu_name = results[0].gpu_name
    run_on_cpu = results[0].run_on_cpu
    assert all(model == r.model for r in results)
    assert all(cpu_name == r.cpu_name for r in results)
    assert not run_on_cpu or gpu_name == ""
    assert all(gpu_name == r.gpu_name for r in results)
    assert all(run_on_cpu == r.run_on_cpu for r in results)
    return model, cpu_name, gpu_name, run_on_cpu

def plot_fitted_line(line, x, y):
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, [m * xv + b for xv in x], color=line.get_color(), linestyle=':')
    plt.text(min(x), max(y), f"t = {m:.4f}n + {b:.4f}", fontsize=11, color=line.get_color())

def format_multisample_data(x: list, y: list) -> Tuple:
    df = pd.DataFrame({'x': x, 'y': y})
    stats = df.groupby('x')['y'].agg(['mean', 'std']).reset_index()
    return (
        np.array(stats['x'].tolist()),
        np.array(stats['mean'].tolist()),
        np.array(stats['std'].tolist())
    )

def plot_cpu_omp_threads_binds_ttft(output_dir: str, results: List[Result]):
    """
    Plot TTFT for varying input sizes.
    Graphs are separated based on the number of concurrent requests.
    """
    model, cpu_name, _, run_on_cpu = get_common_parameters(results)
    if not run_on_cpu:
        return

    output_dir = f"{output_dir}/cpu_omp_threads_bind_ttft"

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
            line, = plt.plot(x, mean, label=cpu_omp_threads_bind, marker='o')
            plt.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2)
            plot_fitted_line(line, x, mean)

        plt.plot([], [], color="grey", label='Fitted line', linestyle=':')
        plt.title(
            f"TTFT vs. CPU OMP Thread Bind\n" \
            f"CPU: {cpu_name}\n" \
            f"Model: {model}, Number of Concurrent Reqs: {num_concurrent_requests}"
        )
        plt.xlabel('Number of Input Tokens')
        plt.ylabel('Time to First Token (seconds)')
        plt.ylim(bottom=0)
        plt.legend()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/cpu_omp_threads_bind_ttft-{num_concurrent_requests}_reqs.png")

def plot_cpu_omp_threads_binds_ttlt(output_dir, results):
    """
    Plot TTFT for varying input sizes.
    Graphs are separated based on the number of concurrent requests.
    """
    model, cpu_name, _, run_on_cpu = get_common_parameters(results)
    if not run_on_cpu:
        return

    output_dir = f"{output_dir}/cpu_omp_threads_bind_ttlt"

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
                    x.append(i+1)
                    y.append(result.time_to_token_s[i])
            x, mean, std = format_multisample_data(x, y)
            line, = plt.plot(x, mean, label=cpu_omp_threads_bind, marker='o')
            plt.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2)
            plot_fitted_line(line, x, mean)

        plt.plot([], [], color="grey", label='Fitted line', linestyle=':')
        plt.title(
            f"Time to Token vs. CPU OMP Thread Bind\n" \
            f"CPU: {cpu_name}\n" \
            f"Model: {model},  Input Tokens: {min_input_tokens},  Concurrent Reqs: {num_concurrent_requests}"
        )
        plt.xlabel('Number of Output Tokens')
        plt.ylabel('Time (seconds)')
        plt.ylim(bottom=0)
        plt.legend()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/cpu_omp_threads_bind_ttlt-{num_concurrent_requests}_reqs.png")

def plot_ttft(output_dir, results):
    """
    Plot time to first token.
    For CPU-runs, the OMP thread bindings with the best top result is used.
    """
    model, cpu_name, gpu_name, run_on_cpu = get_common_parameters(results)

    groups = defaultdict(list)
    for result in results:
        groups[result.num_concurrent_requests].append(result)

    # Identify best OMP thread binds with the best TTFT
    best_ttft_omp_thread_bind_result: Dict[int, Dict[int, Result]] = {}
    for num_concurrent_requests, group in groups.items():
        if num_concurrent_requests not in best_ttft_omp_thread_bind_result:
            best_ttft_omp_thread_bind_result[num_concurrent_requests] = {}

        for result in group:
            if result.num_input_tokens not in best_ttft_omp_thread_bind_result[num_concurrent_requests] or \
            result.time_to_token_s[0] < best_ttft_omp_thread_bind_result[num_concurrent_requests][result.num_input_tokens].time_to_token_s[0]:
                assert num_concurrent_requests == result.num_concurrent_requests
                best_ttft_omp_thread_bind_result[num_concurrent_requests][result.num_input_tokens] = result
    
    # Filter groups to leave only the best TTFT
    best_omp_thread_binds = set()
    for num_concurrent_requests, best_groups in best_ttft_omp_thread_bind_result.items():
        for num_input_tokens, best_result in best_groups.items():
            best_omp_thread_binds.add(best_result.cpu_omp_threads_bind)
            groups[num_concurrent_requests] = list(filter(
                lambda r: r.cpu_omp_threads_bind == best_result.cpu_omp_threads_bind or \
                    r.num_input_tokens != num_input_tokens,
                groups[num_concurrent_requests]
            ))

    # Plot the filtered groups
    plt.figure(figsize=(10, 6))
    for num_reqs, group in groups.items():
        x, y = [], []
        for result in group:
            x.append(result.num_input_tokens)
            y.append(result.time_to_token_s[0])

        x, mean, std = format_multisample_data(x, y)
        line, = plt.plot(x, mean, label=f"Num requests: {num_reqs}", marker='o')
        plt.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2)
        plot_fitted_line(line, x, mean)

    plt.plot([], [], color="grey", label='Fitted line', linestyle=':')
    plt.title(
        f"TTFT vs. Input Token Length\n" \
        f"CPU: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ""}" + "\n" \
        f"Model: {model}" + f"{f', (best results from) OMP Thread Binds: {best_omp_thread_binds}' if run_on_cpu else ""}"
    )
    plt.xlabel('Number of Input Tokens')
    plt.ylabel('Time (seconds)')
    plt.ylim(bottom=0)
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/ttft.png")


def plot_ttlt(output_dir, results):
    """
    Plot time to last token.
    For CPU-runs, the OMP thread bindings with the best top result is used.
    """
    model, cpu_name, gpu_name, run_on_cpu = get_common_parameters(results)

    min_input_tokens = min([r.num_input_tokens for r in results])
    results = list(filter(lambda r: r.num_input_tokens == min_input_tokens, results))

    groups = defaultdict(list)
    for result in results:
        groups[result.num_concurrent_requests].append(result)

    # Identify best OMP thread binds with the best times
    best_ttft_omp_thread_bind_result: Dict[int, Dict[int, Result]] = {}
    for num_concurrent_requests, group in groups.items():
        if num_concurrent_requests not in best_ttft_omp_thread_bind_result:
            best_ttft_omp_thread_bind_result[num_concurrent_requests] = {}

        for result in group:
            if result.num_output_tokens not in best_ttft_omp_thread_bind_result[num_concurrent_requests] or \
            result.time_to_token_s[-1] < best_ttft_omp_thread_bind_result[num_concurrent_requests][result.num_output_tokens].time_to_token_s[-1]:
                assert num_concurrent_requests == result.num_concurrent_requests
                best_ttft_omp_thread_bind_result[num_concurrent_requests][result.num_output_tokens] = result
    
    # Filter groups to leave only the best times
    best_omp_thread_binds = set()
    for num_concurrent_requests, best_groups in best_ttft_omp_thread_bind_result.items():
        for num_output_tokens, best_result in best_groups.items():
            best_omp_thread_binds.add(best_result.cpu_omp_threads_bind)
            groups[num_concurrent_requests] = list(filter(
                lambda r: r.cpu_omp_threads_bind == best_result.cpu_omp_threads_bind or \
                    r.num_output_tokens != num_output_tokens,
                groups[num_concurrent_requests]
            ))

    # Plot the filtered groups
    plt.figure(figsize=(10, 6))
    for num_reqs, group in groups.items():
        x, y = [], []
        for result in group:
            for i in range(result.num_output_tokens):
                x.append(i+1)
                y.append(result.time_to_token_s[i])

        x, mean, std = format_multisample_data(x, y)
        line, = plt.plot(x, mean, label=f"Num requests: {num_reqs}", marker='o')
        plt.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.2)
        plot_fitted_line(line, x, mean)
    
    plt.plot([], [], color="grey", label='Fitted line', linestyle=':')
    plt.title(
        f"Time to Token vs. Output Token Length\n" \
        f"CPU: {cpu_name}" + f"{', GPU: ' + gpu_name if not run_on_cpu else ""}" + "\n" \
        f"Input Tokens: {min_input_tokens}, Model: {model}" + f"{f', (best results from) OMP Thread Binds: {best_omp_thread_binds}' if run_on_cpu else ""}"
    )
    plt.xlabel('Number of Output Tokens')
    plt.ylabel('Time (seconds)')
    plt.ylim(bottom=0)
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/ttlt.png")

if __name__ == "__main__":
    # Load data
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default=None, help='Results directory name')
    args = parser.parse_args()

    if args.name is None:
        # Get latest directory in RESULTS_DIR
        results_dirs = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]
        args.name = max(results_dirs, key=lambda d: os.path.getctime(os.path.join(RESULTS_DIR, d)))

    results_name = args.name
    results = load_csv_data(results_name)

    # Group by model name
    model_groups = defaultdict(list)
    for result in results:
        model_groups[result.model].append(result)

    for model, model_results in model_groups.items():
        output_dir = f"{PLOTS_DIR}/{results_name}/{model}"
        os.makedirs(output_dir, exist_ok=True)
        plot_cpu_omp_threads_binds_ttft(output_dir, model_results)
        plot_cpu_omp_threads_binds_ttlt(output_dir, model_results)
        plot_ttft(output_dir, model_results)
        plot_ttlt(output_dir, model_results)
    

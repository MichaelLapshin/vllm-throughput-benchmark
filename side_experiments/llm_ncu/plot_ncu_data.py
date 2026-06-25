import csv
from typing import Dict, List, Tuple
import subprocess
import io
import csv
import matplotlib.pyplot as plt
import os
import ast
import argparse

from utils import metadata_util

from side_experiments.llm_ncu.csv_headers import H_NUM_OUTPUT_TOKENS, H_NCU_REPORT_DIR, H_NCU_REPORT_FILE
from side_experiments.llm_ncu.constants import (
    RESULTS_PATH, PLOTS_PATH,
    SCHEDULER_LABELS, SCHEDULER_COLOURS,
)
from side_experiments.llm_ncu.speculative_vllm_schedulers import NoSpecDecScheduler_Sequential, NoSpecDecScheduler_Batched

metric_units = {}
time_metric = "gpu__time_duration.sum"
kernel_split_metrics = {
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "pcie__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__instruction_throughput.avg.pct_of_peak_sustained_active",
    "sm__instruction_throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__bytes_read.sum",
    "dram__cycles_active_read.sum",
    "dram__cycles_active_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_active",
    "dram__throughput.avg.peak_sustained",
    "dram__throughput.avg.peak_sustained_active",
    "dram__throughput.avg.per_second",
}

def get_metrics(metadata) -> list:
    ncu_metrics = ast.literal_eval(metadata["parameters"]["NCU_METRICS"])
    ncu_metric_extensions = ast.literal_eval(metadata["parameters"]["NCU_METRIC_EXTENSIONS"])
    metrics = [base + ext for base in ncu_metrics for ext in ncu_metric_extensions] + [time_metric]
    return metrics

def standardize_metric_unit(unit: str, value: float) -> Tuple[str, float]:
    match unit[0]:
        case 'K':
            return unit.removeprefix('K'), value * 1000
        case 'M':
            return unit.removeprefix('M'), value * 1000 * 1000
        case 'G':
            return unit.removeprefix('G'), value * 1000 * 1000 * 1000
        case _:
            return unit, value
    

def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None

def parse_ncu_json(file_path, metrics):
    metrics_arg = ",".join(metrics)
    cmd = ["ncu", "--import", file_path, "--page", "raw", "--print-units", "base", "--csv", "--metrics", metrics_arg]
    # print("Processing: ", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    reader = csv.DictReader(io.StringIO(result.stdout))

    """
    ==PROF== Profiling "unrolled_elementwise_kernel" - 93: 0%....50%....100% - 1 pass
    ==PROF== Profiling "graph" - 94: 0%....50%....100% - 1 pass
    ==PROF== Profiling "unrolled_elementwise_kernel" - 95: 0%....50%....100% - 1 pass
    ==PROF== Profiling "index_elementwise_kernel" - 96: 0%....50%....100% - 1 pass
    ==PROF== Profiling "kernel" - 97: 0%....50%....100% - 1 pass
    ==PROF== Profiling "unrolled_elementwise_kernel" - 98: 0%....50%....100% - 1 pass
    ==PROF== Profiling "reduce_kernel" - 99: 0%....50%....100% - 1 pass
    ==PROF== Profiling "unrolled_elementwise_kernel" - 100: 0%....50%....100% - 1 pass
    """

    metric_data: Dict[str, float] = {}
    kernel_metric_data: Dict[str, Dict[str, Dict[str, float]]] = {
        metric: {} for metric in kernel_split_metrics
    }

    for row in reader:
        kernel_name = row.get("Kernel Name")

        if not kernel_name:
            for metric in metrics:
                unit = row.get(metric)
                if unit:
                    metric_units[metric] = unit
            continue

        time_value = parse_float(row.get(time_metric))
        if time_value is None:
            print("Warning: time_value is None")
            continue

        for metric in metrics:
            value = parse_float(row.get(metric))
            if value is None:
                continue

            unit = metric_units.get(metric, "")
            if unit:
                unit, value = standardize_metric_unit(unit, value)

            metric_data.setdefault(metric, 0.0)
            metric_data[metric] += value

            if metric in kernel_split_metrics:
                kernel_stats = kernel_metric_data[metric].setdefault(
                    kernel_name,
                    {"time_ns": 0.0, "weighted_sum": 0.0},
                )
                kernel_stats["time_ns"] += time_value
                kernel_stats["weighted_sum"] += value * time_value

    # Compute weighted average of kernels
    weighted_average = {}
    for metric in kernel_split_metrics:
        kernel_metrics = kernel_metric_data.get(metric, {})
        total_time = 0.0
        total_weighted = 0.0
        for stats in kernel_metrics.values():
            time_ns = stats.get("time_ns", 0.0)
            total_time += time_ns
            total_weighted += stats.get("weighted_sum", 0.0)

        if total_time <= 0:
            continue

        weighted_average[metric] = total_weighted / total_time

    return {
        "metrics": metric_data,
        "kernel_metrics": kernel_metric_data,
        "weighted_average": weighted_average,
    }
    

def load_report_data(model, results_dir, metadata):
    schedulers_to_test = ast.literal_eval(metadata["parameters"]["SCHEDULERS_TO_TEST"])
    benchmark_output_tokens = ast.literal_eval(metadata["parameters"]["BENCHMARK_OUTPUT_TOKENS"])

    report_data: dict = {} # scheduler -> num_tokens -> metric -> value

    for scheduler in schedulers_to_test:
        report_data[scheduler] = {}

        with open(f"{results_dir}/{model}/{scheduler}/ncu_report_file_mapping.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                num_output_tokens = int(row[H_NUM_OUTPUT_TOKENS])
                report_dir = row[H_NCU_REPORT_DIR]
                report_file = row[H_NCU_REPORT_FILE]

                profile_data = parse_ncu_json(f"{report_dir}/{report_file}", get_metrics(metadata))

                report_data[scheduler][num_output_tokens] = profile_data

                # Filter out data not in benchmark output tokens
                for k in list(report_data[scheduler].keys()):
                    if k not in benchmark_output_tokens:
                        report_data[scheduler].pop(k)

    return report_data

def plot_metrics(model, report_data, metadata, plots_path):
    metrics = get_metrics(metadata)
    schedulers_to_test = ast.literal_eval(metadata["parameters"]["SCHEDULERS_TO_TEST"])

    for metric in metrics:
        plt.figure(figsize=(12, 6))

        if metric not in metric_units:
            print(f"Skipping metric '{metric}'")
            continue

        for scheduler in schedulers_to_test:
            x, y = [], []
            for num_tokens, data in sorted(list(report_data[scheduler].items())):
                metrics = data.get("metrics", data)
                if metric not in metrics:
                    print(f"Strange... skipping '{metric}' for {num_tokens} tokens")
                    continue
                x.append(num_tokens)
                y.append(metrics[metric])
        
            plt.plot(x, y, marker='o', linestyle='--', label=SCHEDULER_LABELS[scheduler], color=SCHEDULER_COLOURS[scheduler], markersize=2)

        plt.title(f"{metric} ({model})", pad=10)
        plt.xlabel("N")
        plt.ylabel(metric_units.get(metric, metric))
        plt.ylim(bottom=0)
        plt.legend()
        plt.tight_layout()
        
        plt.grid(True)
        metrics_path = f"{plots_path}/{model}/ncu_metrics"
        os.makedirs(metrics_path, exist_ok=True)
        plt.savefig(f"{metrics_path}/ncu_{metric}.png", dpi=300)
        plt.close()

def plot_sm_instructions_per_cycle(model, report_data, metadata, plots_path):
    schedulers_to_test = ast.literal_eval(metadata["parameters"]["SCHEDULERS_TO_TEST"])

    plt.figure(figsize=(12, 6))
    for scheduler in schedulers_to_test:
            x, y = [], []
            for num_tokens, data in sorted(list(report_data[scheduler].items())):
                metrics = data.get("metrics", data)
                if "sm__inst_executed.sum" not in metrics:
                    print(f"Strange SM... skipping 'sm__inst_executed.sum' for {num_tokens} tokens")
                    continue
                if "sm__cycles_active.sum" not in metrics:
                    print(f"Strange SM... skipping 'sm__cycles_active.sum' for {num_tokens} tokens")
                    continue
                x.append(num_tokens)
                y.append(metrics["sm__inst_executed.sum"]/float(metrics["sm__cycles_active.sum"]))
        
            plt.plot(x, y, marker='o', linestyle='--', label=SCHEDULER_LABELS[scheduler], color=SCHEDULER_COLOURS[scheduler], markersize=2)


    plt.title(f"SM Instructions Per Active Cycle ({model})", pad=10)
    plt.xlabel("N")
    plt.ylabel("Instructions Per Active Cycle")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    
    plt.grid(True)
    os.makedirs(f"{plots_path}/{model}", exist_ok=True)
    plt.savefig(f"{plots_path}/{model}/ncu_sm_instructions_per_active_cycle.png", dpi=300)
    plt.close()

def sanitize_filename(value: str) -> str:
    safe = []
    for ch in value:
        if ch.isalnum() or ch in "-_+.":
            safe.append(ch)
        else:
            safe.append("_")
    return "".join(safe)[:160]

def plot_kernel_metrics(model, report_data, metadata, plots_path):
    schedulers_to_test = ast.literal_eval(metadata["parameters"]["SCHEDULERS_TO_TEST"])

    kernels_by_metric: Dict[str, set[str]] = {m: set() for m in kernel_split_metrics}
    for scheduler in schedulers_to_test:
        for _, data in report_data[scheduler].items():
            kernel_metrics = data.get("kernel_metrics", {})
            for metric, kernel_data in kernel_metrics.items():
                kernels_by_metric.setdefault(metric, set()).update(kernel_data.keys())

    for metric, kernels in kernels_by_metric.items():
        if not kernels:
            continue
        for kernel_name in sorted(kernels):
            fig, (ax_time, ax_metric) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

            for scheduler in schedulers_to_test:
                x, total_time_ms, weighted_avg = [], [], []
                for num_tokens, data in sorted(report_data[scheduler].items()):
                    kernel_metrics = data.get("kernel_metrics", {}).get(metric, {})
                    stats = kernel_metrics.get(kernel_name)
                    if not stats:
                        continue
                    time_ns = stats["time_ns"]
                    if time_ns <= 0:
                        continue
                    x.append(num_tokens)
                    total_time_ms.append(time_ns / 1_000_000)
                    weighted_avg.append(stats["weighted_sum"] / time_ns)

                if x:
                    ax_time.plot(
                        x,
                        total_time_ms,
                        marker='o',
                        linestyle='--',
                        label=SCHEDULER_LABELS[scheduler],
                        color=SCHEDULER_COLOURS[scheduler],
                        markersize=2,
                    )
                    ax_metric.plot(
                        x,
                        weighted_avg,
                        marker='o',
                        linestyle='--',
                        label=SCHEDULER_LABELS[scheduler],
                        color=SCHEDULER_COLOURS[scheduler],
                        markersize=2,
                    )

            ax_time.set_title(f"Kernel Time\n{kernel_name}")
            ax_time.set_ylabel("Total Kernel Time (ms)")
            ax_time.grid(True)
            ax_time.legend()

            ax_metric.set_title(f"{metric} (weighted avg)")
            ax_metric.set_xlabel("N")
            ax_metric.set_ylabel(metric_units.get(metric, metric))
            ax_metric.grid(True)
            ax_metric.legend()

            kernel_dir = f"{plots_path}/{model}/ncu_metrics_by_kernel/{metric}"
            os.makedirs(kernel_dir, exist_ok=True)
            plt.tight_layout()
            
            plt.savefig(
                f"{kernel_dir}/ncu_{sanitize_filename(kernel_name)}.png",
                dpi=300,
            )
            plt.close()

def plot_weighted_metric_overall(model, report_data, metadata, plots_path):
    schedulers_to_test = ast.literal_eval(metadata["parameters"]["SCHEDULERS_TO_TEST"])

    for metric in kernel_split_metrics:
        plt.figure(figsize=(12, 5))

        for scheduler in schedulers_to_test:
            x, y = [], []
            for num_tokens, data in sorted(report_data[scheduler].items()):
                weighted_average = data.get("weighted_average", {}).get(metric, None)
                if weighted_average:
                    x.append(num_tokens)
                    y.append(weighted_average)

            if x:
                plt.plot(
                    x,
                    y,
                    marker='o',
                    linestyle='--',
                    label=SCHEDULER_LABELS[scheduler],
                    color=SCHEDULER_COLOURS[scheduler],
                    markersize=2,
                )

        plt.title(f"{metric} (weighted avg, overall) ({model})", pad=10)
        plt.xlabel("N")
        plt.ylabel(metric_units.get(metric, metric))
        plt.ylim(bottom=0)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        metrics_path = f"{plots_path}/{model}/ncu_metrics"
        os.makedirs(metrics_path, exist_ok=True)
        plt.savefig(f"{metrics_path}/ncu_{metric}_weighted_overall.png", dpi=300)
        plt.close()

def plot_model_vs_throughput_pct(models, report_data: dict, plots_path):
    N = 1

    pct_metrics_to_profile = [
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__instruction_throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ]

    for metric in pct_metrics_to_profile:
        categories = []
        values = []
        for model in models:
            data = report_data[model]["NoSpecDecScheduler_Sequential"][N]
            weighted_average = data.get("weighted_average", {}).get(metric, None)
            if weighted_average is not None:
                categories.append(model)
                values.append(weighted_average)

        plt.figure(figsize=(10, 5))
        plt.bar(categories, values)
        plt.xticks(rotation=30, ha='right')
        plt.title(f"{metric} (weighted average),  {N} Output Token{'s' if N > 1 else ''}", pad=10)
        plt.xlabel("Model")
        plt.ylabel(metric_units.get(metric, metric))
        plt.ylim(bottom=0)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        metrics_path = f"{plots_path}/comparison"
        os.makedirs(metrics_path, exist_ok=True)
        plt.savefig(f"{metrics_path}/ncu_{metric}.png", dpi=300)
        plt.close()

# Run functions
if __name__ == "__main__":
    # Load data
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default=None, help='Results directory name.')
    args = parser.parse_args()

    if args.name is None:
        # Get latest directory in RESULTS_PATH
        results_dirs = [d for d in os.listdir(RESULTS_PATH) if os.path.isdir(os.path.join(RESULTS_PATH, d))]
        args.name = max(results_dirs, key=lambda d: os.path.getctime(os.path.join(RESULTS_PATH, d)))

    results_dir = f"{RESULTS_PATH}/{args.name}"
    
    # Fetch metadata
    metadata = metadata_util.load_metadata(results_dir)
    models = ast.literal_eval(metadata["parameters"]["MODELS"])

    # Get report data
    report_data = {}
    for model in models:
        report_data[model] = load_report_data(model, results_dir, metadata)

    # Plot
    plots_path = results_dir
    plot_model_vs_throughput_pct(models, report_data, plots_path)
    
    for model in models:
        print(f"Plotting for model: {model}")
        plot_metrics(model, report_data[model], metadata, plots_path)
        plot_sm_instructions_per_cycle(model, report_data[model], metadata, plots_path)
        plot_kernel_metrics(model, report_data[model], metadata, plots_path)
        plot_weighted_metric_overall(model, report_data[model], metadata, plots_path)

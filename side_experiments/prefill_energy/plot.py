import argparse
import os
from typing import Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils.plot_utils import (
    plot_fitted_line, get_poly_colour_no_alpha
)
from run_constants import PROJECT_DIR

RESULTS_DIR = f"{PROJECT_DIR}/side_experiments/prefill_energy/results"


def assert_consistent_num_requests(df):
    df = pd.read_csv(csv_path)
    total_requests = df["num_requests_per_batch"] * df["num_request_batches"]
    assert (total_requests == total_requests.iloc[0]).all(), (
        "num_requests_per_batch * num_request_batches must be constant"
    )

def format_multisample_data(x: list, y: list) -> Tuple:
    df = pd.DataFrame({'x': x, 'y': y})
    stats = df.groupby('x')['y'].agg(['mean', 'std']).reset_index()
    return (
        np.array(stats['x'].tolist()),
        np.array(stats['mean'].tolist()),
        np.array(stats['std'].tolist())
    )

def find_latest_results_csv(results_dir: str) -> str:
    subdirs = [
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No results directories found in {results_dir}")
    latest = max(subdirs, key=lambda d: os.path.getctime(os.path.join(results_dir, d)))
    csv_path = os.path.join(results_dir, latest, "out.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing out.csv at {csv_path}")
    return csv_path


def plot_joules_per_token_by_max_batched_tokens(df, output_path: str) -> None:
    df["num_requests_per_batch"] = pd.to_numeric(df["num_requests_per_batch"], errors="coerce")
    batch_values = sorted(df["num_requests_per_batch"].dropna().unique().tolist())

    if "max_num_batched_tokens" not in df.columns or "joules" not in df.columns:
        raise ValueError("CSV missing required columns: max_num_batched_tokens or joules")

    df["joules_per_token"] = df["joules"] / (
        (df["input_tokens_per_request"] + df["output_tokens_per_request"]) * df["num_requests_per_batch"] * df["num_request_batches"]
    )

    plt.figure(figsize=(10, 6))

    for num_batches in batch_values:
        label = f"num_requests_per_batch={num_batches}"
        subset = df[df["num_requests_per_batch"] == num_batches]
        if subset.empty:
            continue
        x = subset["max_num_batched_tokens"].to_list()
        y = subset["joules_per_token"].to_list()
        x, mean, std = format_multisample_data(x, y)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        plt.plot(x, mean, marker='o', label=label)

    total_requests = df["num_requests_per_batch"] * df["num_request_batches"]
    num_input_tokens = df["input_tokens_per_request"].iloc[0]
    num_output_tokens = df["output_tokens_per_request"].iloc[0]
    plt.title("Joules Per Token vs. Max Num Batched Tokens\n" \
              "(batches executed sequentially)\n" \
              f"Num total requests: {total_requests.iloc[0]}, Input tokens per req: {num_input_tokens}, Output tokens per req: {num_output_tokens}")
    plt.xlabel("Max Num Batched Tokens")
    plt.ylabel("Joules Per Token")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}/joules_per_token_vs_max_batched_tokens.png")
    plt.close()


def plot_throughput_vs_max_batched_tokens(df, output_path: str):
    df["num_requests_per_batch"] = pd.to_numeric(df["num_requests_per_batch"], errors="coerce")
    batch_values = sorted(df["num_requests_per_batch"].dropna().unique().tolist())

    if "time_start_s" not in df.columns or "time_end_s" not in df.columns:
        raise ValueError("CSV missing time_start_s or time_end_s")

    df = df.copy()
    df["exec_time_s"] = df["time_end_s"] - df["time_start_s"]
    df["throughput_tps"] = df["input_tokens_per_request"] * df["num_requests_per_batch"] * df["num_request_batches"] / df["exec_time_s"]

    plt.figure(figsize=(10, 6))
    for num_batches in batch_values:
        label = f"num_requests_per_batch={num_batches}"
        subset = df[df["num_requests_per_batch"] == num_batches]
        if subset.empty:
            continue
        x = subset["max_num_batched_tokens"].to_list()
        y = subset["throughput_tps"].to_list()
        x, mean, std = format_multisample_data(x, y)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)
        plt.plot(x, mean, marker="o", label=label)

    total_requests = df["num_requests_per_batch"] * df["num_request_batches"]
    num_input_tokens = df["input_tokens_per_request"].iloc[0]
    num_output_tokens = df["output_tokens_per_request"].iloc[0]
    plt.title(f"Throughput vs. Max Num Batched Tokens\n" \
              "(batches executed sequentially)\n" \
              f"Num total requests: {total_requests.iloc[0]}, Input tokens per req: {num_input_tokens}, Output tokens per req: {num_output_tokens}")
    plt.xlabel("Max Num Batched Tokens")
    plt.ylabel("Throughput (tokens per second)")
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}/throughput_vs_max_batched_tokens.png")
    plt.close()

def plot_throughput_vs_num_requests_per_batch(df, output_path: str):
    df["num_requests_per_batch"] = pd.to_numeric(df["num_requests_per_batch"], errors="coerce")
    df["max_num_batched_tokens"] = pd.to_numeric(df["max_num_batched_tokens"], errors="coerce")

    max_batched_tokens = df["max_num_batched_tokens"].max()
    total_requests = (df["num_requests_per_batch"] * df["num_request_batches"]).iloc[0]
    df = df[df["max_num_batched_tokens"] == max_batched_tokens]

    if "time_start_s" not in df.columns or "time_end_s" not in df.columns:
        raise ValueError("CSV missing time_start_s or time_end_s")

    df = df.copy()
    df["exec_time_s"] = df["time_end_s"] - df["time_start_s"]
    df["throughput_tps"] = df["input_tokens_per_request"] * df["num_requests_per_batch"] * df["num_request_batches"] / df["exec_time_s"]

    plt.figure(figsize=(10, 6))
    x = df["num_requests_per_batch"].to_list()
    y = df["throughput_tps"].to_list()
    x, mean, std = format_multisample_data(x, y)
    plt.plot(x, mean, marker="o")

    num_input_tokens = df["input_tokens_per_request"].iloc[0]
    num_output_tokens = df["output_tokens_per_request"].iloc[0]
    plt.title(f"Throughput vs. Batch Size\n" \
              "(batches executed sequentially)\n" \
              f"Num total requests: {total_requests}, Input tokens per req: {num_input_tokens}, Output tokens per req: {num_output_tokens}")
    plt.xlabel("Requests Per Batch")
    plt.ylabel("Throughput (tokens per second)")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.savefig(f"{output_path}/throughput_vs_num_requests_per_batch.png")
    plt.close()

def plot_joules_per_token_vs_num_requests_per_batch(df, output_path: str) -> None:
    df["num_requests_per_batch"] = pd.to_numeric(df["num_requests_per_batch"], errors="coerce")
    df["max_num_batched_tokens"] = pd.to_numeric(df["max_num_batched_tokens"], errors="coerce")

    max_batched_tokens = df["max_num_batched_tokens"].max()
    df = df[df["max_num_batched_tokens"] == max_batched_tokens]

    if "max_num_batched_tokens" not in df.columns or "joules" not in df.columns:
        raise ValueError("CSV missing required columns: max_num_batched_tokens or joules")

    df["joules_per_token"] = df["joules"] / (
        (df["input_tokens_per_request"] + df["output_tokens_per_request"]) * df["num_requests_per_batch"] * df["num_request_batches"]
    )

    plt.figure(figsize=(10, 6))
    x = df["num_requests_per_batch"].to_list()
    y = df["joules_per_token"].to_list()
    x, mean, std = format_multisample_data(x, y)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)
    plt.plot(x, mean, marker="o")

    total_requests = df["num_requests_per_batch"] * df["num_request_batches"]
    num_input_tokens = df["input_tokens_per_request"].iloc[0]
    num_output_tokens = df["output_tokens_per_request"].iloc[0]
    plt.title(f"Joules Per Token vs. Requests Per Batch\n" \
              "(batches executed sequentially)\n" \
              f"Num total requests: {total_requests.iloc[0]}, Input tokens per req: {num_input_tokens}, Output tokens per req: {num_output_tokens}")
    plt.xlabel("Requests Per Batch")
    plt.ylabel("Joules Per Token")
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.savefig(f"{output_path}/joules_per_token_vs_num_requests_per_batch.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to out.csv. If omitted, use the latest results in results/",
    )
    args = parser.parse_args()

    csv_path = args.csv or find_latest_results_csv(RESULTS_DIR)
    out_dir = str(Path(csv_path).parent)
    df = pd.read_csv(csv_path)
    assert_consistent_num_requests(df.copy())
    plot_joules_per_token_by_max_batched_tokens(df, out_dir)
    plot_throughput_vs_max_batched_tokens(df.copy(), out_dir)
    plot_throughput_vs_num_requests_per_batch(df.copy(), out_dir)
    plot_joules_per_token_vs_num_requests_per_batch(df.copy(), out_dir)

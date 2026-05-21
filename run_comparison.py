import os
import argparse
from typing import List
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from results import RequestData
from run_constants import RESULTS_DIR, PLOTS_DIR
from utils import metadata_util, plot_utils

@dataclass
class ResultsDir:
    data: List[RequestData]
    metadata: dict

def plot_model_vs_prefill_throughput(output_dir: str, results_dirs: List[ResultsDir]):
    # TODO: find largest common number of input tokens
    pass

def plot_model_vs_decode_throughput(output_dir: str, results_dirs: List[ResultsDir]):
    # TODO: largest common number of output tokens (for 1 input token)
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--names',
        nargs="+", type=str, default=None,
        help='Name of all result directories to compare. Default: All'
    )
    args = parser.parse_args()

    # Get results list
    if args.names is None:
        args.names = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]

    # Load data
    results: List[ResultsDir] = []
    for name in args.names:
        results.append(ResultsDir(
            data=plot_utils.load_csv_data(name),
            metadata=plot_utils.load_metadata(name),
        ))

    # Create dir
    time_now_str = str(datetime.now(ZoneInfo('America/New_York')))
    output_dir = f"{PLOTS_DIR}/comparison-{time_now_str}"
    os.makedirs(output_dir, exist_ok=True)

    # Output
    metadata_util.save_metadata(output_dir, {
        "results_names": args.names,
        "summary": {
            "models": list(set(d.model for r in results for d in r.data)),
            "cpus": list(set(d["CPU_NAME"] for r in results for d in r.metadata)),
            "gpus": list(set(d["GPU_NAME"] for r in results for d in r.metadata)),
        }
    })
    plot_model_vs_prefill_throughput(output_dir, results)
    plot_model_vs_decode_throughput(output_dir, results)

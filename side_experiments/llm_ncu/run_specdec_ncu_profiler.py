import csv
import subprocess
import os
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

from side_experiments.llm_ncu.speculative_vllm_schedulers import SchedulerBase, ProfilerType
from side_experiments.llm_ncu.csv_headers import H_NUM_OUTPUT_TOKENS, H_NCU_REPORT_DIR, H_NCU_REPORT_FILE
from side_experiments.llm_ncu import common_config
from side_experiments.llm_ncu.common_config import (
    RESULTS_PATH,
    BENCHMARK_OUTPUT_TOKENS,
    MODELS,
    SCHEDULERS_TO_TEST,
    NCU_METRICS,
    PROFILE_GPU
)
from utils import metadata_util

# Request to send
def run_benchmark(results_dir):
    metadata_util.save_metadata(
        results_dir,
        data={
            "config": {k: str(v) for k, v in vars(common_config).items() if k.isupper()},
        }
    )

    for model in MODELS:
        for scheduler in SCHEDULERS_TO_TEST:
            report_dir = f"{results_dir}/{model}/{scheduler.__name__}/ncu_profiles"
            os.makedirs(report_dir, exist_ok=True)

            data = {}
            for num_output_tokens in BENCHMARK_OUTPUT_TOKENS:
                report_name = f"tokens_{num_output_tokens}"

                if PROFILE_GPU:
                    command = [
                        "ncu",
                        "--metrics", ",".join(NCU_METRICS),
                        "--nvtx", "--nvtx-include", f"{SchedulerBase.NVTX_PROFILE_NAME}",
                        "--target-processes", "all",
                        "--graph-profiling", "graph",
                        "-o", f"{report_dir}/{report_name}",
                        "-f",
                        "python", "-m", "side_experiments.llm_ncu.launch_scheduler_run_calibrated_request",
                            "-m", model,
                            "-n", f"{num_output_tokens}",
                            "-s", scheduler.__name__,
                            "-p", ProfilerType.NCU_PROFILER.value if PROFILE_GPU else ProfilerType.ITTAPI_PROFILER.value,
                    ]
                else:
                    command = [
                        "vtune",
                        "-collect", "memory-access",
                        "-start-paused",
                        "-result-dir", report_dir,
                        "--", "python", "-m", "side_experiments.llm_ncu.launch_scheduler_run_calibrated_request",
                            "-m", model,
                            "-n", f"{num_output_tokens}",
                            "-s", scheduler.__name__,
                            "-p", ProfilerType.NCU_PROFILER.value if PROFILE_GPU else ProfilerType.ITTAPI_PROFILER.value,
                    ]

                print("Running command \n", " ".join(command))

                # Run the request
                result = subprocess.run(command, check=True)

                # Output data to CSV
                data[num_output_tokens] = report_name

            # Save results to CSV file
            with open(f"{results_dir}/{model}/{scheduler.__name__}/ncu_report_file_mapping.csv", "w") as f:
                fieldnames = (H_NUM_OUTPUT_TOKENS, H_NCU_REPORT_DIR, H_NCU_REPORT_FILE)
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            
                for num_output_tokens, report_name in data.items():
                    writer.writerow({
                        H_NUM_OUTPUT_TOKENS: num_output_tokens,
                        H_NCU_REPORT_DIR: report_dir,
                        H_NCU_REPORT_FILE: f"{report_name}.ncu-rep"
                    })

if __name__ == "__main__":
    load_dotenv()
    timestamp = datetime.now(ZoneInfo('America/New_York'))
    results_dir = f"{RESULTS_PATH}/{timestamp}"
    run_benchmark(results_dir)


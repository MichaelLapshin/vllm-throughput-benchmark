import csv
import subprocess
import os

from side_experiments.llm_ncu.speculative_vllm_schedulers import SchedulerBase
from side_experiments.llm_ncu.csv_headers import H_NUM_OUTPUT_TOKENS, H_NCU_REPORT_DIR, H_NCU_REPORT_FILE
from side_experiments.llm_ncu.common_config import (
    RESULTS_PATH,
    BENCHMARK_OUTPUT_TOKENS,
    MODELS,
    SCHEDULERS_TO_TEST,
    NCU_METRICS
)

# Request to send
def run_benchmark():
    for model in MODELS:
        for scheduler in SCHEDULERS_TO_TEST:
            report_dir = f"{RESULTS_PATH}/{model}/{scheduler.__name__}/ncu_profiles"
            os.makedirs(report_dir, exist_ok=True)

            data = {}
            for num_output_tokens in BENCHMARK_OUTPUT_TOKENS:
                report_name = f"tokens_{num_output_tokens}"

                ncu_command = [
                    "ncu",
                    "--metrics", ",".join(NCU_METRICS),
                    "--nvtx", "--nvtx-include", f"{SchedulerBase.NVTX_PROFILE_NAME}",
                    "--target-processes", "all",
                    "--graph-profiling", "graph",
                    "-o", f"{report_dir}/{report_name}",
                    "-f",
                    "python", "-m", "side_experiments.llm_ncu.benchmarks.launch_scheduler_run_calibrated_request",
                        "-m", model,
                        "-n", f"{num_output_tokens}",
                        "-s", scheduler.__name__,
                ]

                print("Running command \n", " ".join(ncu_command))

                # Run the request
                result = subprocess.run(ncu_command, check=True)

                # Output data to CSV
                data[num_output_tokens] = report_name

            # Save results to CSV file
            with open(f"{RESULTS_PATH}/{model}/{scheduler.__name__}/ncu_report_file_mapping.csv", "w") as f:
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
    run_benchmark()


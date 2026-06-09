import csv
import subprocess
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

from side_experiments.llm_ncu.speculative_vllm_schedulers import SchedulerBase, ProfilerType
from side_experiments.llm_ncu.csv_headers import H_NUM_OUTPUT_TOKENS, H_NCU_REPORT_DIR, H_NCU_REPORT_FILE
from side_experiments.llm_ncu import constants
from side_experiments.llm_ncu import parameters
from side_experiments.llm_ncu.constants import (
    RESULTS_PATH,
)
from side_experiments.llm_ncu.parameters import (
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
            "parameters": {k: str(v) for k, v in vars(parameters).items() if k.isupper()},
            "constants": {k: str(v) for k, v in vars(constants).items() if k.isupper()},
        }
    )

    for model in MODELS:
        for scheduler in SCHEDULERS_TO_TEST:
            report_dir = f"{results_dir}/{model}/{scheduler.__name__}/ncu_profiles"
            os.makedirs(report_dir, exist_ok=True)

            metadata_util.save_metadata(
                report_dir,
                data={"report_dir": report_dir},
            )

            for num_output_tokens in BENCHMARK_OUTPUT_TOKENS:
                report_name = f"tokens_{num_output_tokens}"
                perf_fifo_ctl_path = f"{report_dir}/{report_name}_perf.ctl"
                commands = []

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
                            "-p", ProfilerType.NCU_PROFILER.value,
                    ]

                    metadata_util.add_metadata(
                        report_dir,
                        f"command-token_{num_output_tokens}",
                        command
                    )

                    print("Running command \n", " ".join(command))
                    subprocess.run(command, check=True)
                    commands.append(command)

                    # Save results to CSV file
                    with open(f"{results_dir}/{model}/{scheduler.__name__}/ncu_report_file_mapping.csv", "a") as f:
                        fieldnames = (H_NUM_OUTPUT_TOKENS, H_NCU_REPORT_DIR, H_NCU_REPORT_FILE)
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        if f.tell() == 0:
                            writer.writeheader()
                        writer.writerow({
                            H_NUM_OUTPUT_TOKENS: num_output_tokens,
                            H_NCU_REPORT_DIR: report_dir,
                            H_NCU_REPORT_FILE: f"{report_name}.ncu-rep"
                        })
                else:
                    python_command = [
                        "--",
                        "python", "-m", "side_experiments.llm_ncu.launch_scheduler_run_calibrated_request",
                            "-m", model,
                            "-n", f"{num_output_tokens}",
                            "-s", scheduler.__name__,
                            "-p", ProfilerType.PERF_PROFILER.value,
                            "-c", perf_fifo_ctl_path,
                    ]

                    metadata_util.add_metadata(
                        report_dir,
                        f"python_command-token_{num_output_tokens}",
                        python_command
                    )

                    # Create pipeline
                    subprocess.run(["mkfifo", perf_fifo_ctl_path], check=True)

                    perf_stat_groups = {
                        "default": [],
                        "general": [
                            "-e", "cycles",
                            "-e", "instructions",
                            "-M", "cpu_operating_frequency",
                        ],
                        "uncore_imc": [
                            "-e", "uncore_imc/cas_count_read/",
                            "-e", "uncore_imc/cas_count_write/",
                        ],
                        "memory": [
                            "-e", "mem-loads",
                            "-e", "mem-stores",
                        ],
                        "bandwidth": [
                            "-M", "cpu_utilization",
                            "-M", "memory_bandwidth_read",
                            "-M", "memory_bandwidth_write",
                            "-M", "loads_per_instr",
                        ],
                        "tma_backend_bound_group": [
                            "-M", "tma_backend_bound_group",
                        ],
                        "tma_memory_bound_group": [
                            "-M", "tma_memory_bound_group",
                        ],
                        "tma_l1": [
                            "-M", "TmaL1",
                        ],
                        "tma_l2": [
                            "-M", "TmaL2",
                        ],
                        "tma_l3_mem": [
                            "-M", "TmaL3mem",
                        ],
                        "top_down_l5": [
                            "-M", "TopdownL5",
                        ],
                        "energy" : [
                            "-e", "power/energy-pkg/",
                            "-e", "power/energy-ram/",
                        ],
                    }

                    metadata_util.add_metadata(
                        report_dir,
                        f"python_command-token_{num_output_tokens}",
                        python_command
                    )

                    # Run perf command
                    for rec_intervals in [False, True]:
                        perf_stat_command = [
                            "perf", "stat",
                            "--delay=-1", # start the perf perofiler as paused
                            f"--control", f"fifo:{perf_fifo_ctl_path}",
                            # "-A", # do not aggregate counts across all monitored CPUs
                            "-j", # print output in json format'
                            "-a",
                        ]

                        if rec_intervals:
                            perf_stat_command += ["-I", "1"]
                        else:
                            perf_stat_command += ["-r", "3"] # take multiple samples

                        metadata_util.add_metadata(report_dir, f"perf_stat_command{'-interval_1ms' if rec_intervals else ''}", perf_stat_command)
                        metadata_util.add_metadata(report_dir, f"perf_stat_groups{'-interval_1ms' if rec_intervals else ''}", perf_stat_groups)

                        for stat_group_name, stat_group_args in perf_stat_groups.items():
                            metadata_util.add_metadata(
                                report_dir,
                                f"stat_group_name-token_{num_output_tokens}-{stat_group_name}",
                                stat_group_args
                            )
                            
                            command = perf_stat_command + stat_group_args + python_command
                            print("Running command \n", " ".join(command))
                            perf_result = subprocess.run(
                                command,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=True,
                            )

                            with open(f"{report_dir}/{report_name}_perf_stat-{stat_group_name}{'-interval_1ms' if rec_intervals else ''}.data", "a") as f:
                                enabled = False if rec_intervals else True
                                for line in perf_result.stderr.splitlines():
                                    if rec_intervals:
                                        if "Events enabled" in line:
                                            enabled = True
                                            continue
                                        
                                        if "Events disabled" in line:
                                            enabled = False
                                            continue

                                    if enabled:
                                        try:
                                            line_json = json.loads(line)
                                            pcnt_running = line_json.get("pcnt-running", None)
                                            assert pcnt_running is None or pcnt_running == 100.00, \
                                                f"Must have perfect perf recording. {line_json=}"
                                        except Exception as e:
                                            continue
                                        f.write(line + "\n")

                    # Perf mem
                    perf_mem_command = [
                        "perf", "mem", "record",
                        "--delay=-1", # start the perf perofiler as paused
                        f"--control", f"fifo:{perf_fifo_ctl_path}",
                        "-o", f"{report_dir}/{report_name}_perf_mem.data",
                    ]

                    subprocess.run(
                        perf_mem_command + python_command,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True,
                    )   

                    metadata_util.add_metadata(
                        report_dir,
                        f"perf_mem_command-token_{num_output_tokens}",
                        perf_mem_command
                    )
                    
                    os.remove(perf_fifo_ctl_path) # cleanup

if __name__ == "__main__":
    load_dotenv()
    timestamp = datetime.now(ZoneInfo('America/New_York'))
    results_dir = f"{RESULTS_PATH}/{timestamp}"
    run_benchmark(results_dir)


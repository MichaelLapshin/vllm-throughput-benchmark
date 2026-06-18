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
    PROFILE_GPU,
    PERF_STAT_RUNS, PERF_STAT_PROFILE_METRICS,
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
        for scheduler_name in SCHEDULERS_TO_TEST:
            report_dir = f"{results_dir}/{model}/{scheduler_name}/ncu_profiles"
            os.makedirs(report_dir, exist_ok=True)

            metadata_util.save_metadata(
                report_dir,
                data={"report_dir": report_dir},
            )

            for num_output_tokens in BENCHMARK_OUTPUT_TOKENS:
                report_name = f"tokens_{num_output_tokens}"
                perf_fifo_ctl_path = f"{report_dir}/{report_name}_perf.ctl"
                perf_fifo_ack_path = f"{report_dir}/{report_name}_perf.ack"
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
                            "-s", scheduler_name,
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
                    with open(f"{results_dir}/{model}/{scheduler_name}/ncu_report_file_mapping.csv", "a") as f:
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
                            "-s", scheduler_name,
                            "-p", ProfilerType.PERF_PROFILER.value,
                            "-c", perf_fifo_ctl_path,
                            "-a", perf_fifo_ack_path,
                    ]

                    metadata_util.add_metadata(
                        report_dir,
                        f"python_command-token_{num_output_tokens}",
                        python_command
                    )

                    # Create pipeline
                    subprocess.run(["mkfifo", perf_fifo_ctl_path], check=True)
                    subprocess.run(["mkfifo", perf_fifo_ack_path], check=True)

                    perf_stat_groups = {
                        "default": [],
                        "ipc": [
                            "-e", "cpu-cycles",
                            "-e", "instructions",
                        ],
                        "uncore_imc": [
                            "-e", "uncore_imc/cas_count_read/",
                            "-e", "uncore_imc/cas_count_write/",
                            "-e", "uncore_imc/clockticks/",
                        ],
                        "cache" : [
                            "-e", "cache-misses",
                            "-e", "cache-references",
                        ],
                        "l1_cache": [
                            "-e", "L1-dcache-loads",
                            "-e", "L1-dcache-load-misses",
                            "-e", "L1-dcache-stores",
                            "-e", "L1-icache-load-misses",
                        ],
                        "llc_cache_load": [
                            "-e", "LLC-loads",
                            "-e", "LLC-load-misses",
                        ],
                        "llc_cache_store": [
                            "-e", "LLC-stores",
                            "-e", "LLC-store-misses",
                        ],
                        "topdown_bubbles": [
                            "-e", "topdown-fetch-bubbles",
                            "-e", "topdown-recovery-bubbles",
                            "-e", "topdown-total-slots",
                        ],
                        "topdown_slots": [
                            "-e", "topdown-slots-issued",
                            "-e", "topdown-slots-retired",
                            "-e", "topdown-total-slots",
                        ],
                        "cycle_activity_l1d": [
                            "-e", "cycle_activity.cycles_l1d_pending",
                            "-e", "cycle_activity.stalls_l1d_pending",
                        ],
                        "cycle_activity_l2": [
                            "-e", "cycle_activity.cycles_l2_pending",
                            "-e", "cycle_activity.stalls_l2_pending",
                        ],
                        "cycle_activity_ldm": [
                            "-e", "cycle_activity.cycles_ldm_pending",
                            "-e", "cycle_activity.stalls_ldm_pending",
                        ],
                        "resource_stalls": [
                            "-e", "resource_stalls.any",
                            "-e", "resource_stalls.rob",
                            "-e", "resource_stalls.rs",
                            "-e", "resource_stalls.sb",
                        ],
                        "uncore_memory": [
                            "-e", "llc_misses.mem_read",
                            "-e", "llc_misses.mem_write",
                            "-e", "unc_m_cas_count.rd",
                            "-e", "unc_m_cas_count.wr",
                        ],
                        "energy" : [
                            "-e", "power/energy-pkg/",
                            "-e", "power/energy-ram/",
                        ],
                        "memory": [
                            "-e", "mem-loads",
                            "-e", "mem-stores",
                        ],
                    }

                    if PERF_STAT_PROFILE_METRICS:
                        perf_stat_groups = perf_stat_groups | {
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
                            f"--control", f"fifo:{perf_fifo_ctl_path},{perf_fifo_ack_path}",
                            # "-A", # do not aggregate counts across all monitored CPUs
                            "-j", # print output in json format'
                            "-a",
                        ]

                        if rec_intervals:
                            perf_stat_command += ["-I", "1"]
                        else:
                            perf_stat_command += ["-r", f"{PERF_STAT_RUNS}"] # take multiple samples

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
                                        pcnt_running = None
                                        try:
                                            line_json = json.loads(line)
                                            pcnt_running = line_json.get("pcnt-running", None)
                                        except Exception as e:
                                            continue
                                        
                                        if pcnt_running is not None and pcnt_running != 100.00:
                                            # assert False, f"Must have perfect perf recording. {line_json=}"
                                            f.write(f"Warning: pcnt_running ({pcnt_running}) is not 100%\n")
                                        f.write(line + "\n")

                    # Perf mem
                    perf_mem_command = [
                        "perf", "mem", "record",
                        "--delay=-1", # start the perf perofiler as paused
                        f"--control", f"fifo:{perf_fifo_ctl_path},{perf_fifo_ack_path}",
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
                    
                    # Cleanup
                    os.remove(perf_fifo_ctl_path) 
                    os.remove(perf_fifo_ack_path)

if __name__ == "__main__":
    load_dotenv()
    timestamp = datetime.now(ZoneInfo('America/New_York'))
    results_dir = f"{RESULTS_PATH}/{timestamp}"
    run_benchmark(results_dir)


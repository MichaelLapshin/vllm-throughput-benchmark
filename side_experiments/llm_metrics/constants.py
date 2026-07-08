import pathlib
import os

from side_experiments.llm_metrics.speculative_vllm_schedulers import (
    NoSpecDecScheduler_Sequential, NoSpecDecScheduler_Batched, NoSpecDecScheduler_Batched_16ot,
)

# Paths
EXPERIMENT_PATH = f"{pathlib.Path(__file__).parent.resolve()}"

RESULTS_PATH = f"{EXPERIMENT_PATH}/results"
os.makedirs(RESULTS_PATH, exist_ok=True)

PLOTS_PATH = f"{EXPERIMENT_PATH}/plots"
os.makedirs(PLOTS_PATH, exist_ok=True)

# Labels
SCHEDULER_LABELS = {
    NoSpecDecScheduler_Sequential.__name__: "Sequential Decoding (N tokens x 1 request)",
    NoSpecDecScheduler_Batched.__name__: "Batched Decoding (1 token x N requests)",
    NoSpecDecScheduler_Batched_16ot.__name__: "Batched Decoding (16 tokens x N requests)"
}

SCHEDULER_COLOURS = {
    NoSpecDecScheduler_Sequential.__name__: "navy",
    NoSpecDecScheduler_Batched.__name__: "orange",
    NoSpecDecScheduler_Batched_16ot.__name__: "green",
}

# vLLM environment variables
os.environ["VLLM_CONFIGURE_LOGGING"] = "1"
os.environ["VLLM_LOGGING_CONFIG_PATH"] = f"{EXPERIMENT_PATH}/vllm_logging.json"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1" # Switch between `InprocClient` and `MPClient`

# vLLM constant parameters
GPU_MEMORY_UTILIZATION=0.90

# Benchmark environment variables
ENABLE_PREDICT_BONUS_TOKEN = False
os.environ["ENABLE_PREDICT_BONUS_TOKEN"] = "true" if ENABLE_PREDICT_BONUS_TOKEN else "false"

# NCU metrics
COMPUTE_THROUGHPUT_METRICS = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    # Submetrics
    "sm__pipe_tensor_cycles_active_v2.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_tex.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed.avg.pct_of_peak_sustained_elapsed",
    "idc__request_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__memory_throughput_internal_activity.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_ipa.avg.pct_of_peak_sustained_elapsed",
    "sm__mio_inst_issued.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_adu.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
    "sm__instruction_throughput_internal_activity.avg.pct_of_peak_sustained_elapsed",
    "sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_cbu_pred_on_any.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_elapsed",
    "sm__mio2rf_writeback_active.avg.pct_of_peak_sustained_elapsed",
    "sm__mio_pq_read_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_fmaheavy_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_elapsed",
]

MEMORY_THROUGHPUT_METRICS = [
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    # Submetrics
    "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l1tex__data_pipe_tex_wavefronts.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_access_throughput_internal_activity.avg.pct_of_peak_sustained_elapsed",
    "fbpa__dram_sectors.avg.pct_of_peak_sustained_elapsed",
    "l1tex__m_xbar2l1tex_read_sectors.avg.pct_of_peak_sustained_elapsed",
    "l1tex__data_bank_writes.avg.pct_of_peak_sustained_elapsed",
    "l1tex__texin_sm2tex_req_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "lts__d_sectors_fill_device.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_request_throughput_internal_activity.avg.pct_of_peak_sustained_elapsed",
    "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
    "lts__xbar2lts_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l1tex__m_l1tex2xbar_req_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l1tex__lsuin_requests.avg.pct_of_peak_sustained_elapsed",
    "lts__d_sectors.avg.pct_of_peak_sustained_elapsed",
    "l1tex__data_bank_reads.avg.pct_of_peak_sustained_elapsed",
    "lts__d_sectors_fill_sysmem.avg.pct_of_peak_sustained_elapsed",
    "l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed",
    "lts__t_tag_requests.avg.pct_of_peak_sustained_elapsed",
    "l1tex__tex_writeback_active.avg.pct_of_peak_sustained_elapsed",
    "lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_elapsed",
    "lts__lts2xbar_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "l1tex__f_wavefronts.avg.pct_of_peak_sustained_elapsed",
    # Extras
    "gpu__compute_memory_access_throughput.max.pct_of_peak_sustained_active",
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed",
]

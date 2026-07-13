PERF_STAT_GROUPS = {
    "default": [],
    "ipc": [
        "-e", "cpu-cycles",
        "-e", "instructions",
    ],
    "uncore_imc": [
        "-e", "uncore_imc_0/cas_count_read/",
        "-e", "uncore_imc_1/cas_count_read/",
        "-e", "uncore_imc_2/cas_count_read/",
        "-e", "uncore_imc_3/cas_count_read/",
        "-e", "uncore_imc_0/cas_count_write/",
        "-e", "uncore_imc_1/cas_count_write/",
        "-e", "uncore_imc_2/cas_count_write/",
        "-e", "uncore_imc_3/cas_count_write/",
        "-e", "uncore_imc_0/clockticks/",
        "-e", "uncore_imc_1/clockticks/",
        "-e", "uncore_imc_2/clockticks/",
        "-e", "uncore_imc_3/clockticks/",
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

PERF_STAT_METRIC_GROUPS = {
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
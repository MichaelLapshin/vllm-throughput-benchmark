from dataclasses import dataclass
from typing import List

@dataclass
class RequestData:
    # Parameters
    request_batch_uid: str
    model: str
    cpu_omp_threads_bind: str
    num_warmup_runs: int
    num_runs: int
    run_num: int
    num_concurrent_requests: int
    num_input_tokens: int
    num_output_tokens: int

    # State
    cpu_temp_before_run: int
    cpu_temp_after_run: int
    gpu_temp_before_run: int
    gpu_temp_after_run: int

    # Metrics
    queued_ts: float
    scheduled_ts: float
    first_token_ts: float
    last_token_ts: float

    # Output
    time_to_token_s: List[float]
    request_batch_energy_joules: int

    @staticmethod
    def from_dict(data: dict) -> 'RequestData':
        return RequestData(**data)
    
@dataclass
class EmissionsData:
    timestamp: str
    project_name: str
    run_id: str
    experiment_id: str
    duration: float
    emissions: float
    emissions_rate: float
    cpu_power: float
    gpu_power: float
    ram_power: float
    cpu_energy: float
    gpu_energy: float
    ram_energy: float
    energy_consumed: float
    water_consumed: float
    country_name: str
    country_iso_code: str
    region: str
    cloud_provider: str
    cloud_region: str
    os: str
    python_version: str
    codecarbon_version: str
    cpu_count: int
    cpu_model: str
    gpu_count: int
    gpu_model: str
    longitude: float
    latitude: float
    ram_total_size: float
    tracking_mode: str
    cpu_utilization_percent: float
    gpu_utilization_percent: float
    ram_utilization_percent: float
    ram_used_gb: float
    on_cloud: str
    pue: float
    wue: float

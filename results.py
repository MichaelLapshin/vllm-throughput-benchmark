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
    

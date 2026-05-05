from dataclasses import dataclass
from typing import List

@dataclass
class Result:
    # Parameters
    request_batch_uid: str
    model: str
    cpu_name: str
    gpu_name: str 
    run_on_cpu: bool
    cpu_omp_threads_bind: str
    num_warmup_samples: int
    num_samples: int
    num_concurrent_requests: int
    num_input_tokens: int
    num_output_tokens: int

    # Metrics
    queued_ts: float
    scheduled_ts: float
    first_token_ts: float
    last_token_ts: float

    # Output
    time_to_token_s: List[float]

    @staticmethod
    def from_dict(data: dict) -> 'Result':
        return Result(**data)
    

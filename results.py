from dataclasses import dataclass
from typing import List

@dataclass
class Result:
    # Parameters
    model: str
    cpu_name: str
    gpu_name: str 
    run_on_cpu: bool
    cpu_omp_threads_bind: List[int]
    num_warmup_samples: int
    num_samples: int
    num_input_tokens: int
    num_output_tokens: int

    # Results
    prefill_time_s: float
    decode_token_times_s: List[float]

    @staticmethod
    def from_dict(data: dict) -> 'Result':
        return Result(**data)
    

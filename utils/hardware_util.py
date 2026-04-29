import cpuinfo
import torch

def get_cpu_name() -> str:
    return cpuinfo.get_cpu_info()["brand_raw"]

def get_gpu_name(gpu_number: int) -> str:
    return torch.cuda.get_device_name(0)

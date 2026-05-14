import cpuinfo
import torch
import psutil
from pynvml import *

_cpu_name = None
_gpu_names = {}
_gpu_devices = {}

def _get_gpu_device(gpu_number: int):
    global _gpu_devices
    if gpu_number not in _gpu_devices:
        _gpu_devices[gpu_number] = nvmlDeviceGetHandleByIndex(gpu_number)
    return _gpu_devices[gpu_number]

def get_cpu_name() -> str:
    global _cpu_name
    if _cpu_name is None:
        _cpu_name = cpuinfo.get_cpu_info()["brand_raw"]
    return _cpu_name

def get_gpu_name(gpu_number: int) -> str:
    global _gpu_name
    if gpu_number not in _gpu_names:
        _gpu_names[gpu_number] = torch.cuda.get_device_name(gpu_number)
    return _gpu_names[gpu_number]

def get_cpu_cores_avg_temp() -> int:
    temps = psutil.sensors_temperatures()
    temps_sum = []
    for entry in temps['coretemp']:
        if 'coretemp' in entry:
            temps_sum.append(entry.current)
    return -1 if not temps_sum else int(sum(temps_sum)/float(len(temps_sum)))

def get_gpu_temp(gpu_number: int) -> int:
    return nvmlDeviceGetTemperature(_get_gpu_device(gpu_number), 0)
    
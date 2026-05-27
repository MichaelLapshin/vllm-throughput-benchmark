import cpuinfo
import torch
import psutil
import pynvml
from typing import List
import time

_nvml_initialized = False
_gpu_count = None
_cpu_name = None
_gpu_names = {}
_gpu_devices = {}

def initialize_nvml():
    global _nvml_initialized
    if not _nvml_initialized:
        pynvml.nvmlInit()
        _nvml_initialized = True

def shutdown_nvml():
    global _nvml_initialized
    if _nvml_initialized:
        pynvml.nvmlShutdown()
        _nvml_initialized = False

def _get_gpu_device(gpu_number: int):
    initialize_nvml()
    global _gpu_devices
    if gpu_number not in _gpu_devices:
        _gpu_devices[gpu_number] = pynvml.nvmlDeviceGetHandleByIndex(gpu_number)
    return _gpu_devices[gpu_number]

def get_cpu_name() -> str:
    global _cpu_name
    if _cpu_name is None:
        _cpu_name = cpuinfo.get_cpu_info()["brand_raw"]
    return _cpu_name

def get_gpu_name(gpu_number: int) -> str:
    initialize_nvml()
    global _gpu_name
    if gpu_number not in _gpu_names:
        _gpu_names[gpu_number] = torch.cuda.get_device_name(gpu_number)
    return _gpu_names[gpu_number]

def get_cpu_cores_avg_temp() -> int:
    temps = psutil.sensors_temperatures()
    temps_sum = []
    for entry in temps.get('coretemp', []):
        if 'coretemp' in entry:
            temps_sum.append(entry.current)
    return -1 if not temps_sum else int(sum(temps_sum)/float(len(temps_sum)))

def get_gpu_temp(gpu_number: int) -> int:
    initialize_nvml()
    return pynvml.nvmlDeviceGetTemperature(_get_gpu_device(gpu_number), 0)

def get_gpu_count() -> int:
    initialize_nvml()
    global _gpu_count
    if _gpu_count is None:
        _gpu_count = pynvml.nvmlDeviceGetCount()
    return _gpu_count

def get_gpu_names() -> List[str]:
    initialize_nvml()
    gpu_count = get_gpu_count()
    return [get_gpu_name(i) for i in range(gpu_count)]

def get_gpu_energy_consumption_joules(gpu_number: int) -> float:
    # Returns current energy in joules
    initialize_nvml()
    return pynvml.nvmlDeviceGetTotalEnergyConsumption(_get_gpu_device(gpu_number)) / 1000.0

def get_gpu_power_watts(gpu_number: int) -> float:
    # Returns current energy in watts
    initialize_nvml()
    return pynvml.nvmlDeviceGetPowerUsage(_get_gpu_device(gpu_number)) / 1000.0

def get_gpu_freq_mhz(gpu_number: int, clock) -> int:
    return pynvml.nvmlDeviceGetClockInfo(_get_gpu_device(gpu_number), clock)
    
def can_read_intel_rapl_energy() -> bool:
    try:
        with open('/sys/class/powercap/intel-rapl:0/energy_uj', 'r') as f:
            energy_uj = int(f.read().strip())
        return True
    except Exception:
        return False
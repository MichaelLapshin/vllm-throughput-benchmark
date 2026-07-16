import cpuinfo
import torch
import psutil
import pynvml
from typing import List
import time
import os
import glob

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

def set_cpu_max_frequency(khz: int = 0):
    # Find all active CPU policy directories
    policies = glob.glob("/sys/devices/system/cpu/cpu*/cpufreq")

    if not policies:
        print("Error: CPU frequency scaling interface not found.")
        return

    for policy in policies:
        max_path = os.path.join(policy, "scaling_max_freq")

        # Read frequency boundaries
        cpuinfo_min_freq_path = os.path.join(policy, "cpuinfo_min_freq")
        with open(cpuinfo_min_freq_path, "r") as f:
            cpuinfo_min_freq = int(f.readline())

        cpuinfo_max_freq_path = os.path.join(policy, "cpuinfo_max_freq")
        with open(cpuinfo_max_freq_path, "r") as f:
            cpuinfo_max_freq = int(f.readline())
        
        assert cpuinfo_min_freq <= khz <= cpuinfo_max_freq

        # Set target frequency
        target_khz = cpuinfo_max_freq if khz == 0 else str(khz)

        with open(max_path, "r") as f:
            if int(f.readline()) == target_khz:
                continue # don't change frequency is already set to target

        # Set and verify the CPU's frequency
        with open(max_path, "w") as f:
            f.write(str(target_khz))
        
        with open(max_path, "r") as f:
            assert int(f.readline()) == target_khz
    
    print(f"Successfully updated all CPU ranges to max {khz / 1000:.1f} MHz")

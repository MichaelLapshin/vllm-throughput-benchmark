import psutil
from typing import List
from utils import process_util

CPU_THREADS_ENV_VAR="VLLM_CPU_OMP_THREADS_BIND"


def get_vllm_process_logic_threads(cmd_regex: str) -> List[int]:
    """
    Find the logic threads used to start the process by analyzing 
    the environment variable VLLM_CPU_OMP_THREADS_BIND.
    
    NOTE(mlapshin): This is useful. https://docs.vllm.ai/en/stable/getting_started/installation/cpu/#build-image-from-source

    :param process_command: Regex of the vllm command.
    :type process_command: str
    :return: String parameter for the threads
    :rtype: The list of threads used to start the vLLM process.
    """

    vllm_proc: psutil.Process = process_util.get_proc(cmd_regex)
    prov_env = vllm_proc.environ()
    
    if CPU_THREADS_ENV_VAR not in prov_env:
        return []

    assert CPU_THREADS_ENV_VAR in prov_env
    threads_str = prov_env[CPU_THREADS_ENV_VAR]    
    return process_util.thread_bind_str_to_list(threads_str)

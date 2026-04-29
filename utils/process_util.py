import re
import psutil
import getpass
from typing import List

def get_proc(cmdline_regex) -> psutil.Process:
    pattern = re.compile(cmdline_regex)

    result = None
    for proc in psutil.process_iter(attrs=["pid", "name", "username", "cmdline"]):
        if len(proc.info["cmdline"]) <= 1:
            continue
        
        if proc.info["username"] != getpass.getuser():
            continue

        if pattern.search(" ".join(proc.info["cmdline"])):
            assert result is None, "Should not match to multiple commands; it's ambiguous"
            result = proc
            
    assert result is not None, "Could not find process using the command"
    return result


def thread_bind_str_to_list(threads_bind_str: str) -> List[int]:
    """
    Converts thread-binding list format to a list.
    Example: "1-3,7-8" -> [1, 2, 3, 7, 8]
    """
    result = []
    for part in threads_bind_str.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            result.extend(range(start, end + 1))
        else:
            result.append(int(part))
    return result

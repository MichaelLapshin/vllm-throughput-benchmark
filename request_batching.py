import os
import tempfile

"""
This lock is for ensuring that requests can be added to the vLLM scheduler in one batch.

Otherwise, there would be a race-condition where if requests are sequentially added, then
the earlier-added requests may be scheduled before the others are added.

Whenever `_batch_count` is greater than 0, vLLM will wait until that number of
requests are added before proceeding with input processing. 
"""

BATCH_COUNT_FILE = os.path.join(tempfile.gettempdir(), "vllm_batch_count.temp.txt")

def set_batch_count(count: int):
    with open(BATCH_COUNT_FILE, 'w') as f:
        f.write(str(count))

def get_batch_count() -> int:
    try:
        if os.path.exists(BATCH_COUNT_FILE):
            with open(BATCH_COUNT_FILE, 'r') as f:
                return int(f.read())
    except:
        pass
    return 0           

def decrement_batch_count_to_zero() -> bool:
    """
    @return True, if batch count was decremented.
    """
    val = get_batch_count()
    if val > 0:
        set_batch_count(val - 1)
        return True
    return False
        
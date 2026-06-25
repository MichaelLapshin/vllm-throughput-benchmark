import os
import shutil
import time
import struct
import fcntl
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional, Any

VLLM_TMP_LOCK_DIR = "/tmp/vllm_ipc_locks"

def clear_tmp_lock_dir():
    """
    Clear the temporary lock directory.
    """
    shutil.rmtree(VLLM_TMP_LOCK_DIR)

@dataclass
class _FileLock:
    """Interprocess lock via flock() on a lockfile (Linux/Unix)."""
    path: str

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Open each time so the lock works even if processes fork/spawn.
        self._fd = open(self.path, "a+")
        fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fd is not None:
            try:
                fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
            finally:
                self._fd.close()
                self._fd = None


class SharedMemoryValue:
    """
    Value-like object backed by named SharedMemory + a filesystem lock.
    Interface: .value, .get_lock().
    """
    def __init__(
        self,
        name: str,
        fmt: str = "q",              # int64 by default
        create: bool = False,
        lock_dir: str = "/tmp/vllm_ipc_locks",
        init_value: Any = 0,
    ):
        self.name = name
        self.fmt = fmt
        self.size = struct.calcsize(fmt)
        self._lock = _FileLock(os.path.join(lock_dir, f"{name}.lock"))

        # Create-or-attach
        try:
            self._shm = shared_memory.SharedMemory(name=name, create=create, size=self.size)
            created = create
        except FileExistsError:
            self._shm = shared_memory.SharedMemory(name=name, create=False, size=self.size)
            created = False

        # Initialize once (best-effort; guarded by lock)
        if created:
            with self.get_lock():
                self.value = init_value

    def get_lock(self):
        return self._lock

    @property
    def value(self):
        (v,) = struct.unpack(self.fmt, self._shm.buf[: self.size])
        return v

    @value.setter
    def value(self, v):
        self._shm.buf[: self.size] = struct.pack(self.fmt, v)

    def close(self):
        self._shm.close()

    def unlink(self):
        """Only call from the owner/creator process once you're sure nobody else is using it."""
        self._shm.unlink()


class SharedMemoryEvent:
    """
    Event-like object backed by named SharedMemory + a filesystem lock.
    Interface: .set(), .clear(), .is_set(), .wait(timeout), .get_lock().
    """
    def __init__(
        self,
        name: str,
        create: bool = False,
        lock_dir: str = VLLM_TMP_LOCK_DIR,
        poll_interval_s: float = 0.001,
    ):
        self._flag = SharedMemoryValue(
            name=name,
            fmt="b",          # signed char
            create=create,
            lock_dir=lock_dir,
            init_value=0,
        )
        self._poll_interval_s = poll_interval_s

    def get_lock(self):
        return self._flag.get_lock()

    def is_set(self) -> bool:
        return bool(self._flag.value)

    def set(self) -> None:
        with self.get_lock():
            self._flag.value = 1

    def clear(self) -> None:
        with self.get_lock():
            self._flag.value = 0

    def wait(self, timeout: Optional[float] = None) -> bool:
        deadline = None if timeout is None else (time.time() + timeout)
        while True:
            if self.is_set():
                return True
            if deadline is not None and time.time() >= deadline:
                return False
            time.sleep(self._poll_interval_s)

    def close(self):
        self._flag.close()

    def unlink(self):
        self._flag.unlink()
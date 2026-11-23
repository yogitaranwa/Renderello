"""Fail-closed sampled resident-memory monitor, no optional dependencies.

Windows WorkingSetSize is resident bytes, not commit or peak allocation.
This is a polling guard, not an OS allocation ceiling or process-tree limit.
"""
import ctypes
import sys
from pathlib import Path


class ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_uint32), ('PageFaultCount', ctypes.c_uint32)] + [
        (name, ctypes.c_size_t) for name in (
            'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
            'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


def _windows_api():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int32, ctypes.c_uint32]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int32
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters), ctypes.c_uint32]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int32
    return kernel, psapi


class ResidentMemory:
    def __init__(self, pid, *, platform=None, windows_api=None):
        if type(pid) is not int or not 0 < pid <= 0xffffffff:
            raise ValueError('positive DWORD process id required')
        self.pid = pid
        self.platform = sys.platform if platform is None else platform
        self.handle = None
        self.closed = False
        if self.platform == 'win32':
            self.kernel, self.psapi = (windows_api or _windows_api)()
            # Query only, non-inherited. Modern Windows supports limited query access.
            self.handle = self.kernel.OpenProcess(0x1000, False, pid)
            if not self.handle:
                raise RuntimeError('Windows process memory handle unavailable')
        elif not self.platform.startswith('linux'):
            raise RuntimeError('resident-memory monitor unsupported on this platform')

    def sample(self):
        if self.closed:
            raise RuntimeError('resident-memory monitor is closed')
        if self.platform == 'win32':
            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            if not self.psapi.GetProcessMemoryInfo(self.handle, ctypes.byref(counters), counters.cb):
                raise RuntimeError('Windows process memory sample unavailable')
            value = counters.WorkingSetSize
        else:
            text = Path(f'/proc/{self.pid}/status').read_text()
            line = next((line for line in text.splitlines() if line.startswith('VmRSS:')), None)
            fields = line.split() if line else []
            if len(fields) != 3 or fields[2] != 'kB':
                raise RuntimeError('Linux resident-memory sample unavailable')
            value = int(fields[1]) * 1024
        if value <= 0:
            raise RuntimeError('invalid resident-memory sample')
        return value

    def close(self):
        if not self.closed:
            self.closed = True
            if self.handle is not None:
                handle, self.handle = self.handle, None
                if not self.kernel.CloseHandle(handle):
                    raise RuntimeError('Windows process memory handle close failed')

"""Read process and container memory counters without a monitoring service."""
from pathlib import Path
import resource
import sys


def memoryUsage():
    """Return counters with explicit scope; no claim of per-job isolation."""
    scale = 1 if sys.platform == "darwin" else 1024
    data = {
        "processPeakBytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale),
        "scope": "process lifetime peak; container counters cover all processes",
    }
    for label, locations in {
        "containerCurrentBytes": ["/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"],
        "containerPeakBytes": ["/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"],
        "containerLimitBytes": ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"],
    }.items():
        for location in locations:
            try:
                value = Path(location).read_text().strip()
                data[label] = int(value) if value != "max" else None
                break
            except (OSError, ValueError):
                continue
    return data

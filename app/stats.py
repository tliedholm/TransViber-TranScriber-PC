"""Background poller for the system-utilization readout in the status bar.

Ported from the Mac edition; the macOS ioreg GPU query is replaced with
NVML. Polling runs in a QThread because device queries take long enough to
stutter the UI from a main-thread QTimer.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

POLL_MS = 2000

# NVML is initialized once; False means "tried and unavailable" so the label
# degrades to CPU/RAM only (e.g. driver issues) instead of retrying forever.
_nvml: tuple | bool | None = None


def _nvml_handle():
    global _nvml
    if _nvml is None:
        try:
            import pynvml

            pynvml.nvmlInit()
            _nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
        except Exception:
            _nvml = False
    return _nvml if _nvml else None


def gpu_stats() -> tuple[int | None, int | None]:
    """(GPU utilization %, VRAM used %) — whole-system numbers, or Nones."""
    nv = _nvml_handle()
    if nv is None:
        return None, None
    pynvml, handle = nv
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(util), int(mem.used / mem.total * 100)
    except Exception:
        return None, None


class StatsPoller(QThread):
    stats = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        import psutil  # imported here, like every other non-Qt dependency

        psutil.cpu_percent(interval=None)  # prime the since-last-call counter
        while not self._stop:
            parts = []
            gpu, vram = gpu_stats()
            if gpu is not None:
                parts.append(f"GPU {gpu}%")
            if vram is not None:
                parts.append(f"VRAM {vram}%")
            parts.append(f"CPU {psutil.cpu_percent(interval=None):.0f}%")
            parts.append(f"RAM {psutil.virtual_memory().percent:.0f}%")
            self.stats.emit(" · ".join(parts))
            self.msleep(POLL_MS)

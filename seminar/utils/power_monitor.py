import os
import logging
import time
import threading
import numpy as np
import psutil
from config import CPU_TDP_WATTS, CPU_IDLE_WATTS

logger = logging.getLogger(__name__)

# Try to use pynvml for GPU power (faster than subprocess nvidia-smi)
try:
    import pynvml as _pynvml
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False


class PowerMonitor:
    def __init__(self, sample_interval=0.01, target_pid=None, cpu_only=False):
        self.sample_interval = sample_interval
        self.readings = []
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        self._gpu_handle = None
        self.gpu_available = False if cpu_only else self._check_gpu()
        self.target_pid = target_pid or os.getpid()
        self._process = psutil.Process(self.target_pid)
        try:
            self._process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def _check_gpu(self):
        if _HAS_PYNVML:
            try:
                _pynvml.nvmlInit()
                self._gpu_handle = _pynvml.nvmlDeviceGetHandleByIndex(0)
                _pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
                return True
            except Exception as e:
                logger.debug("pynvml GPU detection failed: %s", e)
                self._gpu_handle = None
                return False
        return False

    def _get_gpu_power(self):
        if self._gpu_handle is None:
            return 0.0
        try:
            return _pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle) / 1000.0
        except Exception:
            return 0.0

    def _get_cpu_metrics(self):
        try:
            proc_cpu = self._process.cpu_percent(interval=None)
            sys_cpu = psutil.cpu_percent(interval=None)
            mem_percent = self._process.memory_percent()
            return proc_cpu, sys_cpu, mem_percent
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0.0, 0.0, 0.0

    def _estimate_cpu_power(self, proc_cpu_percent):
        num_cores = psutil.cpu_count(logical=True)
        normalized_percent = min(proc_cpu_percent / num_cores, 100.0)
        power = CPU_IDLE_WATTS + (CPU_TDP_WATTS - CPU_IDLE_WATTS) * (normalized_percent / 100.0)
        return power

    def _sample_loop(self):
        while self._running:
            timestamp = time.perf_counter()
            gpu_power = self._get_gpu_power() if self.gpu_available else 0.0
            proc_cpu, sys_cpu, mem_percent = self._get_cpu_metrics()
            cpu_power = self._estimate_cpu_power(proc_cpu)
            reading = {
                'timestamp': timestamp,
                'gpu_power': gpu_power,
                'cpu_power': cpu_power,
                'proc_cpu_percent': proc_cpu,
                'sys_cpu_percent': sys_cpu,
                'memory_percent': mem_percent
            }
            with self._lock:
                self.readings.append(reading)
            time.sleep(self.sample_interval)

    def start(self):
        with self._lock:
            self.readings = []
        self._running = True
        try:
            self._process.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        with self._lock:
            return list(self.readings)

    def get_energy_stats(self):
        with self._lock:
            readings = list(self.readings)

        if not readings or len(readings) < 2:
            return {
                'avg_power': 0, 'max_power': 0, 'energy_joules': 0,
                'avg_cpu_power': 0, 'avg_cpu_percent': 0, 'max_cpu_percent': 0, 'num_samples': 0
            }

        gpu_powers = [r['gpu_power'] for r in readings]
        cpu_powers = [r['cpu_power'] for r in readings]
        cpu_percents = [r['proc_cpu_percent'] for r in readings]
        timestamps = [r['timestamp'] for r in readings]

        gpu_energy = 0
        cpu_energy = 0
        for i in range(1, len(readings)):
            dt = timestamps[i] - timestamps[i-1]
            gpu_energy += ((gpu_powers[i] + gpu_powers[i-1]) / 2) * dt
            cpu_energy += ((cpu_powers[i] + cpu_powers[i-1]) / 2) * dt

        total_energy = gpu_energy + cpu_energy
        avg_power = np.mean(gpu_powers) + np.mean(cpu_powers) if self.gpu_available else np.mean(cpu_powers)
        max_power = np.max(gpu_powers) + np.max(cpu_powers) if self.gpu_available else np.max(cpu_powers)

        return {
            'avg_power': avg_power,
            'max_power': max_power,
            'energy_joules': total_energy,
            'avg_gpu_power': np.mean(gpu_powers) if gpu_powers else 0,
            'avg_cpu_power': np.mean(cpu_powers) if cpu_powers else 0,
            'avg_cpu_percent': np.mean(cpu_percents) if cpu_percents else 0,
            'max_cpu_percent': np.max(cpu_percents) if cpu_percents else 0,
            'num_samples': len(readings),
            'duration': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        }

import os
import logging
import time
import threading
import subprocess
import numpy as np
import psutil
from config import CPU_TDP_WATTS, CPU_IDLE_WATTS

logger = logging.getLogger(__name__)


class PowerMonitor:
    def __init__(self, sample_interval=0.01, target_pid=None):
        self.sample_interval = sample_interval
        self.readings = []
        self._running = False
        self._thread = None
        self.gpu_available = self._check_gpu()
        self.target_pid = target_pid or os.getpid()
        self._process = psutil.Process(self.target_pid)
        try:
            self._process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def _check_gpu(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                logger.debug("nvidia-smi returned %d", result.returncode)
                return False
            float(result.stdout.strip().split('\n')[0])
            return True
        except FileNotFoundError:
            logger.debug("nvidia-smi not found")
            return False
        except subprocess.TimeoutExpired:
            logger.debug("nvidia-smi timed out")
            return False
        except (ValueError, IndexError) as e:
            logger.debug("GPU detection failed: %s", e)
            return False

    def _get_gpu_power(self):
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            return float(result.stdout.strip().split('\n')[0])
        except (subprocess.TimeoutExpired, ValueError, IndexError, FileNotFoundError):
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
            timestamp = time.time()
            gpu_power = self._get_gpu_power() if self.gpu_available else 0.0
            proc_cpu, sys_cpu, mem_percent = self._get_cpu_metrics()
            cpu_power = self._estimate_cpu_power(proc_cpu)
            self.readings.append({
                'timestamp': timestamp,
                'gpu_power': gpu_power,
                'cpu_power': cpu_power,
                'proc_cpu_percent': proc_cpu,
                'sys_cpu_percent': sys_cpu,
                'memory_percent': mem_percent
            })
            time.sleep(self.sample_interval)

    def start(self):
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
        return self.readings

    def get_energy_stats(self):
        if not self.readings or len(self.readings) < 2:
            return {
                'avg_power': 0, 'max_power': 0, 'energy_joules': 0,
                'avg_cpu_power': 0, 'avg_cpu_percent': 0, 'max_cpu_percent': 0, 'num_samples': 0
            }

        gpu_powers = [r['gpu_power'] for r in self.readings]
        cpu_powers = [r['cpu_power'] for r in self.readings]
        cpu_percents = [r['proc_cpu_percent'] for r in self.readings]
        timestamps = [r['timestamp'] for r in self.readings]

        gpu_energy = 0
        cpu_energy = 0
        for i in range(1, len(self.readings)):
            dt = timestamps[i] - timestamps[i-1]
            gpu_energy += ((gpu_powers[i] + gpu_powers[i-1]) / 2) * dt
            cpu_energy += ((cpu_powers[i] + cpu_powers[i-1]) / 2) * dt

        if self.gpu_available and any(p > 0 for p in gpu_powers):
            avg_power = np.mean(gpu_powers)
            max_power = np.max(gpu_powers)
            total_energy = gpu_energy
        else:
            avg_power = np.mean(cpu_powers)
            max_power = np.max(cpu_powers)
            total_energy = cpu_energy

        return {
            'avg_power': avg_power,
            'max_power': max_power,
            'energy_joules': total_energy,
            'avg_gpu_power': np.mean(gpu_powers) if gpu_powers else 0,
            'avg_cpu_power': np.mean(cpu_powers) if cpu_powers else 0,
            'avg_cpu_percent': np.mean(cpu_percents) if cpu_percents else 0,
            'max_cpu_percent': np.max(cpu_percents) if cpu_percents else 0,
            'num_samples': len(self.readings),
            'duration': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        }

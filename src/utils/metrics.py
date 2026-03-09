import logging
import time
import numpy as np
import torch
import psutil

logger = logging.getLogger(__name__)
CURRENT_PROCESS = psutil.Process()


def measure_energy(predict_fn, input_array, power_monitor, device="cpu",
                   num_reps=10, warmup_reps=5):
    for _ in range(warmup_reps):
        try:
            predict_fn(input_array)
        except (RuntimeError, ValueError) as e:
            logger.warning("Warmup failed: %s", e)
            return {
                'avg_power': 0, 'max_power': 0, 'energy_per_inference': 0,
                'latency': 0.001, 'avg_cpu_percent': 0, 'cpu_time_per_inference': 0
            }

    if device == "cuda":
        torch.cuda.synchronize()

    power_monitor.start()
    cpu_start = CURRENT_PROCESS.cpu_times()
    cpu_start_total = cpu_start.user + cpu_start.system
    start_time = time.perf_counter()

    for _ in range(num_reps):
        try:
            predict_fn(input_array)
        except (RuntimeError, ValueError) as e:
            logger.warning("Inference failed: %s", e)
            power_monitor.stop()
            return {
                'avg_power': 0, 'max_power': 0, 'energy_per_inference': 0,
                'latency': 0.001, 'avg_cpu_percent': 0, 'cpu_time_per_inference': 0
            }

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    cpu_end = CURRENT_PROCESS.cpu_times()
    cpu_end_total = cpu_end.user + cpu_end.system
    power_monitor.stop()
    power_stats = power_monitor.get_energy_stats()

    total_time = end_time - start_time
    total_cpu = cpu_end_total - cpu_start_total

    return {
        'avg_power': power_stats['avg_power'],
        'max_power': power_stats['max_power'],
        'energy_joules': power_stats.get('energy_joules', 0),
        'energy_per_inference': power_stats.get('energy_joules', 0) / num_reps,
        'cpu_time_per_inference': total_cpu / num_reps,
        'latency': total_time / num_reps,
        'avg_cpu_percent': power_stats.get('avg_cpu_percent', 0),
        'max_cpu_percent': power_stats.get('max_cpu_percent', 0),
        'avg_cpu_power': power_stats.get('avg_cpu_power', 0),
    }


def measure_latency(predict_fn, input_array, power_monitor, device="cpu",
                    num_reps=20, warmup_reps=10):
    for _ in range(warmup_reps):
        try:
            predict_fn(input_array)
        except (RuntimeError, ValueError) as e:
            logger.warning("Warmup failed: %s", e)
            return {
                'latency': 0.001, 'latency_std': 0, 'latency_max': 0, 'latency_min': 0,
                'avg_power': 0, 'avg_cpu_percent': 0
            }

    if device == "cuda":
        torch.cuda.synchronize()

    power_monitor.start()
    latencies = []

    for _ in range(num_reps):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            predict_fn(input_array)
        except (RuntimeError, ValueError) as e:
            logger.warning("Inference failed: %s", e)
            power_monitor.stop()
            return {
                'latency': 0.001, 'latency_std': 0, 'latency_max': 0, 'latency_min': 0,
                'avg_power': 0, 'avg_cpu_percent': 0
            }
        if device == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append(end - start)

    power_monitor.stop()
    power_stats = power_monitor.get_energy_stats()

    return {
        'latency': np.mean(latencies),
        'latency_std': np.std(latencies),
        'latency_max': np.max(latencies),
        'latency_min': np.min(latencies),
        'avg_power': power_stats['avg_power'],
        'max_power': power_stats.get('max_power', 0),
        'avg_cpu_percent': power_stats.get('avg_cpu_percent', 0),
        'max_cpu_percent': power_stats.get('max_cpu_percent', 0)
    }

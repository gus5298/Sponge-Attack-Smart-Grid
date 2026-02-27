import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np
import argparse
from chronos import ChronosPipeline

from config import (CONTEXT_LEN, NUM_FEATURES, PREDICTION_LEN,
                    REPS_PER_MEASUREMENT, POPULATION_SIZE, ENERGY_SAMPLE_INTERVAL, OUTPUT_DIR)
from utils.model_loader import load_seed
from utils.power_monitor import PowerMonitor
from utils.metrics import measure_latency
from utils.ga_operators import time_slice_crossover, latency_mutation, create_latency_population
from utils.attack_runner import AttackHistory, create_ga, run_ga, print_results
from utils.visualization import plot_ga_evolution

parser = argparse.ArgumentParser()
parser.add_argument("--generations", type=int, default=50)
parser.add_argument("--mode", type=str, choices=["constrained", "extreme"], default="extreme")
args = parser.parse_args()

print("="*70)
print("CHRONOS LATENCY ATTACK")
print("="*70)

device = "cpu"
power_monitor = PowerMonitor(ENERGY_SAMPLE_INTERVAL, cpu_only=True)
OUT = os.path.join(OUTPUT_DIR, "chronos")

print("\nLoading Chronos model...")
pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base", device_map=device, torch_dtype=torch.float32,
)

seed_data, mean, std = load_seed()
flat_seed = seed_data.flatten()


def make_prediction(input_array):
    if len(input_array.shape) == 2:
        univariate = input_array[:, 0]
    else:
        univariate = input_array
    inp = torch.tensor(univariate, dtype=torch.float32)
    pipeline.predict(inp, prediction_length=PREDICTION_LEN)


print("\nMeasuring baseline...")
baseline_stats = measure_latency(make_prediction, seed_data, power_monitor, device, num_reps=2)
BASELINE_LATENCY = baseline_stats['latency']
BASELINE_POWER = baseline_stats['avg_power']
print(f"Baseline Latency: {BASELINE_LATENCY*1000:.3f}ms")

history = AttackHistory()


def fitness_func(ga_instance, solution, solution_idx):
    input_array = solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    stats = measure_latency(make_prediction, input_array, power_monitor, device, 1)
    latency_ratio = stats['latency'] / BASELINE_LATENCY if BASELINE_LATENCY > 0 else 1.0
    fitness = latency_ratio * 100
    history.record_solution(fitness, solution, latency=stats['latency'],
                            power=stats['avg_power'], cpu_percent=stats['avg_cpu_percent'])
    return float(fitness)


def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    best = history.end_generation(gen, extra_columns={
        'best_latency': 'latency', 'best_power': 'power', 'best_cpu_percent': 'cpu_percent',
        'global_best_fitness': 'fitness'
    })
    if best:
        lat_change = ((best['latency'] / BASELINE_LATENCY) - 1) * 100 if BASELINE_LATENCY > 0 else 0
        print(f"Gen {gen:3d}: Fitness={best['fitness']:.4f} (Latency {lat_change:+.1f}%)")


initial_population, gene_space = create_latency_population(flat_seed, POPULATION_SIZE, args.mode)

ga_instance = create_ga(
    fitness_func=fitness_func, on_generation=on_generation,
    num_genes=len(flat_seed), num_generations=args.generations,
    initial_population=initial_population, gene_space=gene_space,
    crossover_type=time_slice_crossover, mutation_type=latency_mutation,
)

if __name__ == "__main__":
    best_solution, _ = run_ga(ga_instance, "Chronos Latency Attack")

    adv_input = best_solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    verify_adv = measure_latency(make_prediction, adv_input, power_monitor, device, 100)
    verify_base = measure_latency(make_prediction, seed_data, power_monitor, device, 100)

    print_results("CHRONOS LATENCY ATTACK", [
        ("Latency (ms)", verify_base['latency'] * 1000, verify_adv['latency'] * 1000, "{:.3f}"),
    ])

    prefix = "chronos_latency"
    history.save(prefix, OUT)

    plot_ga_evolution(history.generation_data,
                      {'latency': BASELINE_LATENCY, 'power': BASELINE_POWER},
                      [
                          {'data_key': 'global_best_fitness', 'ylabel': 'Fitness', 'title': 'Fitness Evolution'},
                          {'data_key': 'best_latency', 'ylabel': 'Latency (ms)', 'title': 'Latency Evolution', 'scale': 1000, 'baseline_key': 'latency'},
                          {'data_key': 'best_power', 'ylabel': 'Power (W)', 'title': 'Power Evolution', 'color': 'orange'},
                          {'data_key': 'best_cpu_percent', 'ylabel': 'CPU %', 'title': 'CPU Utilization', 'color': 'g'},
                      ],
                      'Chronos Latency Attack Results', os.path.join(OUT, f"{prefix}_results.png"))

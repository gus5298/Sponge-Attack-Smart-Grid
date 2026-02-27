import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import time
import torch
import numpy as np
import argparse

from config import (CONTEXT_LEN, NUM_FEATURES, MAX_PONDERS,
                    POPULATION_SIZE, ENERGY_SAMPLE_INTERVAL, OUTPUT_DIR)
from utils.model_loader import load_act, load_seed_from_csv
from utils.power_monitor import PowerMonitor
from utils.ga_operators import time_slice_crossover, latency_mutation, create_latency_population
from utils.attack_runner import AttackHistory, create_ga, run_ga, print_results
from utils.visualization import plot_ga_evolution

parser = argparse.ArgumentParser()
parser.add_argument("--generations", type=int, default=50)
parser.add_argument("--mode", type=str, choices=["constrained", "extreme"], default="extreme")
args = parser.parse_args()

print("=" * 70)
print("ACT-LSTM LATENCY ATTACK - Maximizing Ponder Steps")
print("=" * 70)
print(f"Goal: Find inputs that trigger {MAX_PONDERS} ponder steps.")

device = "cpu"
power_monitor = PowerMonitor(ENERGY_SAMPLE_INTERVAL, cpu_only=True)
model = load_act(device=device)
seed_data, mean, std = load_seed_from_csv()
flat_seed = seed_data.flatten()
OUT = os.path.join(OUTPUT_DIR, "act")


def measure_inference(input_tensor):
    start = time.perf_counter()
    with torch.no_grad():
        _, avg_ponder = model(input_tensor)
    end = time.perf_counter()
    return end - start, avg_ponder.item()


history = AttackHistory()

# Baseline
print("\nMeasuring baseline...")
print("Warming up...", end="", flush=True)
dummy = torch.randn(1, CONTEXT_LEN, NUM_FEATURES).to(device)
for _ in range(50):
    model(dummy)
print(" Done.")

base_tensor = torch.tensor(seed_data.reshape(1, CONTEXT_LEN, NUM_FEATURES), dtype=torch.float32).to(device)
base_lat, base_ponder = measure_inference(base_tensor)
BASELINE_PONDER = base_ponder
BASELINE_LATENCY = base_lat
print(f"Baseline: Ponders={base_ponder:.2f}/{MAX_PONDERS}, Latency={base_lat*1000:.3f}ms")


def fitness_func(ga_instance, solution, solution_idx):
    input_tensor = torch.tensor(solution.reshape(1, CONTEXT_LEN, NUM_FEATURES), dtype=torch.float32).to(device)
    latency, ponder_steps = measure_inference(input_tensor)
    fitness = float(ponder_steps ** 2)
    history.record_solution(fitness, solution, ponder=ponder_steps, latency=latency)
    return fitness


def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    best = history.end_generation(gen, extra_columns={
        'best_ponder': 'ponder', 'best_latency': 'latency',
        'global_best_fitness': 'fitness'
    })
    if best:
        print(f"Gen {gen:3d}: Ponder={best['ponder']:.2f}/{MAX_PONDERS} | Latency={best['latency']*1000:.3f}ms")


initial_population, gene_space = create_latency_population(flat_seed, POPULATION_SIZE, args.mode)

ga_instance = create_ga(
    fitness_func=fitness_func, on_generation=on_generation,
    num_genes=len(flat_seed), num_generations=args.generations,
    initial_population=initial_population, gene_space=gene_space,
    crossover_type=time_slice_crossover, mutation_type=latency_mutation,
)

if __name__ == "__main__":
    best_solution, _ = run_ga(ga_instance, "ACT-LSTM Latency Attack")

    adv_tensor = torch.tensor(best_solution.reshape(1, CONTEXT_LEN, NUM_FEATURES), dtype=torch.float32).to(device)
    adv_lat, adv_ponder = measure_inference(adv_tensor)

    print_results("ACT-LSTM LATENCY ATTACK", [
        ("Ponder Steps", base_ponder, adv_ponder, "{:.2f}"),
        ("Latency (ms)", base_lat * 1000, adv_lat * 1000, "{:.3f}"),
    ])

    prefix = "act_latency"
    history.save(prefix, OUT)

    plot_ga_evolution(history.generation_data,
                      {'ponder': BASELINE_PONDER, 'latency': BASELINE_LATENCY},
                      [
                          {'data_key': 'global_best_fitness', 'ylabel': 'Fitness', 'title': 'Fitness Evolution'},
                          {'data_key': 'best_ponder', 'ylabel': 'Ponder Steps', 'title': 'Ponder Evolution', 'baseline_key': 'ponder'},
                          {'data_key': 'best_latency', 'ylabel': 'Latency (ms)', 'title': 'Latency Evolution', 'scale': 1000, 'baseline_key': 'latency'},
                      ],
                      'ACT-LSTM Latency Attack Results', os.path.join(OUT, f"{prefix}_results.png"))

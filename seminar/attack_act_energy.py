import torch
import numpy as np

from config import (CONTEXT_LEN, NUM_FEATURES, BASELINE_REPS,
                    POPULATION_SIZE, NUM_PARENTS_MATING)
from utils.model_loader import load_act, load_seed_from_csv, make_predictor, get_device
from utils.power_monitor import PowerMonitor
from utils.metrics import measure_energy
from utils.attack_runner import print_results

import pygad

print("="*70)
print("ACT-LSTM ENERGY ATTACK")
print("="*70)

device = get_device()
power_monitor = PowerMonitor(0.01)

model = load_act(device=device)
seed_data, mean, std = load_seed_from_csv()
flat_seed = seed_data.flatten()
predict = make_predictor(model, device)

print("\nMeasuring baseline...")
baseline_stats = measure_energy(predict, seed_data, power_monitor, device, num_reps=BASELINE_REPS)
BASELINE_ENERGY = baseline_stats['energy_per_inference']
BASELINE_POWER = baseline_stats['avg_power']
print(f"Baseline Power: {BASELINE_POWER:.1f}W")

generation_data = {'gen': [], 'best_energy': [], 'best_power': [], 'best_latency': []}


def fitness_func(ga_instance, solution, solution_idx):
    input_array = solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    stats = measure_energy(predict, input_array, power_monitor, device, BASELINE_REPS)
    return stats['energy_per_inference'] / BASELINE_ENERGY if BASELINE_ENERGY > 0 else 1.0


def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    best_solution, best_fitness, _ = ga_instance.best_solution()
    input_array = best_solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    stats = measure_energy(predict, input_array, power_monitor, device, BASELINE_REPS)
    generation_data['gen'].append(gen)
    generation_data['best_energy'].append(stats['energy_per_inference'])
    generation_data['best_power'].append(stats['avg_power'])
    generation_data['best_latency'].append(stats['latency'])
    print(f"Gen {gen:3d}: Energy={stats['energy_per_inference']*1000:.3f}mJ")


if __name__ == "__main__":
    print("\nStarting ACT-LSTM Energy Attack...")

    print("Warming up...", end="", flush=True)
    dummy = torch.randn(1, CONTEXT_LEN, NUM_FEATURES).to(device)
    for _ in range(20):
        model(dummy)
    print(" Done.")

    initial_population = np.array([
        flat_seed + np.random.normal(0, 0.5, len(flat_seed))
        for _ in range(POPULATION_SIZE)
    ], dtype=np.float32)

    ga_instance = pygad.GA(
        num_generations=50,
        num_parents_mating=NUM_PARENTS_MATING,
        fitness_func=fitness_func,
        sol_per_pop=POPULATION_SIZE,
        num_genes=len(flat_seed),
        initial_population=initial_population,
        on_generation=on_generation,
        suppress_warnings=True
    )

    ga_instance.run()

    best_solution, _, _ = ga_instance.best_solution()
    adv_input = best_solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)

    adv_stats = measure_energy(predict, adv_input, power_monitor, device, BASELINE_REPS)
    base_stats = measure_energy(predict, seed_data, power_monitor, device, BASELINE_REPS)

    print_results("ACT-LSTM ENERGY ATTACK", [
        ("Energy (J)", base_stats['energy_per_inference'], adv_stats['energy_per_inference'], "{:.4f}"),
        ("Power (W)", base_stats['avg_power'], adv_stats['avg_power'], "{:.1f}"),
        ("Latency (ms)", base_stats['latency'] * 1000, adv_stats['latency'] * 1000, "{:.1f}"),
    ])

    np.save("act_energy_best_input.npy", best_solution)

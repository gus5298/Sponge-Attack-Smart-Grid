import time
import numpy as np
import argparse

from config import (CONTEXT_LEN, NUM_FEATURES, REPS_PER_MEASUREMENT, BASELINE_REPS,
                    VERIFICATION_REPS, POPULATION_SIZE, ENERGY_SAMPLE_INTERVAL)
from utils.model_loader import load_deepar, load_seed, make_predictor, get_device
from utils.power_monitor import PowerMonitor
from utils.metrics import measure_energy
from utils.ga_operators import time_slice_crossover, energy_mutation, create_energy_population
from utils.attack_runner import AttackHistory, create_ga, run_ga, print_results
from utils.visualization import plot_ga_evolution

parser = argparse.ArgumentParser()
parser.add_argument("--generations", type=int, default=50)
parser.add_argument("--mode", type=str, choices=["constrained", "extreme"], default="extreme")
args = parser.parse_args()

print("="*70)
print("DeepAR ENERGY ATTACK - Maximizing GPU Power Consumption")
print("="*70)

device = get_device()
power_monitor = PowerMonitor(ENERGY_SAMPLE_INTERVAL)

model, checkpoint = load_deepar(device=device)
seed_data, mean, std = load_seed(checkpoint=checkpoint)
flat_seed = seed_data.flatten()
predict = make_predictor(model, device)

print("\nMeasuring baseline...")
baseline_stats = measure_energy(predict, seed_data, power_monitor, device, num_reps=BASELINE_REPS)
BASELINE_POWER = baseline_stats['avg_power']
BASELINE_ENERGY = baseline_stats['energy_per_inference']
BASELINE_CPU = baseline_stats['cpu_time_per_inference']
BASELINE_LATENCY = baseline_stats['latency']
print(f"Baseline Power: {BASELINE_POWER:.1f}W, Energy: {BASELINE_ENERGY*1000:.3f}mJ")

history = AttackHistory()


def fitness_func(ga_instance, solution, solution_idx):
    input_array = solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    stats = measure_energy(predict, input_array, power_monitor, device, REPS_PER_MEASUREMENT)
    power_ratio = stats['avg_power'] / BASELINE_POWER if BASELINE_POWER > 0 else 1.0
    cpu_ratio = stats['cpu_time_per_inference'] / BASELINE_CPU if BASELINE_CPU > 0 else 1.0
    if power_monitor.gpu_available:
        fitness = 0.7 * power_ratio + 0.3 * cpu_ratio
    else:
        lat_ratio = stats['latency'] / BASELINE_LATENCY if BASELINE_LATENCY > 0 else 1.0
        fitness = 0.6 * cpu_ratio + 0.4 * lat_ratio
    history.record_solution(fitness, solution, power=stats['avg_power'],
                            energy=stats['energy_per_inference'], latency=stats['latency'],
                            cpu_percent=stats['avg_cpu_percent'])
    return float(fitness)


def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    best = history.end_generation(gen, extra_columns={
        'best_power': 'power', 'best_energy': 'energy', 'best_latency': 'latency',
        'best_cpu_percent': 'cpu_percent', 'global_best_fitness': 'fitness'
    })
    if best:
        power_change = ((best['power'] / BASELINE_POWER) - 1) * 100 if BASELINE_POWER > 0 else 0
        print(f"Gen {gen:3d}: Fitness={best['fitness']:.4f} (Power {power_change:+.1f}%)")


initial_population, gene_space = create_energy_population(flat_seed, POPULATION_SIZE, args.mode)

ga_instance = create_ga(
    fitness_func=fitness_func, on_generation=on_generation,
    num_genes=len(flat_seed), num_generations=args.generations,
    initial_population=initial_population, gene_space=gene_space,
    crossover_type=time_slice_crossover, mutation_type=energy_mutation,
)

if __name__ == "__main__":
    best_solution, _ = run_ga(ga_instance, "DeepAR Energy Attack")

    adv_input = best_solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    verify_adv = measure_energy(predict, adv_input, power_monitor, device, VERIFICATION_REPS)
    verify_base = measure_energy(predict, seed_data, power_monitor, device, VERIFICATION_REPS)

    print_results("DeepAR ENERGY ATTACK", [
        ("Power (W)", verify_base['avg_power'], verify_adv['avg_power'], "{:.1f}"),
        ("Energy (mJ)", verify_base['energy_per_inference'] * 1000, verify_adv['energy_per_inference'] * 1000, "{:.3f}"),
    ])

    prefix = "deepar_energy"
    history.save(prefix)

    plot_ga_evolution(history.generation_data,
                      {'power': BASELINE_POWER, 'energy': BASELINE_ENERGY},
                      [
                          {'data_key': 'global_best_fitness', 'ylabel': 'Fitness', 'title': 'Fitness Evolution'},
                          {'data_key': 'best_power', 'ylabel': 'Power (W)', 'title': 'Power Evolution', 'baseline_key': 'power'},
                          {'data_key': 'best_energy', 'ylabel': 'Energy (mJ)', 'title': 'Energy Evolution', 'scale': 1000, 'baseline_key': 'energy'},
                          {'data_key': 'best_cpu_percent', 'ylabel': 'CPU %', 'title': 'CPU Utilization', 'color': 'g'},
                      ],
                      'DeepAR Energy Attack Results', f"{prefix}_results.png")

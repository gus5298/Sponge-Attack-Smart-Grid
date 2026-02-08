import time
import torch
import numpy as np

from config import (CONTEXT_LEN, NUM_FEATURES, MAX_PONDERS,
                    POPULATION_SIZE, NUM_PARENTS_MATING)
from utils.model_loader import load_act, load_seed_from_csv, get_device
from utils.power_monitor import PowerMonitor
from utils.attack_runner import AttackHistory
from utils.visualization import plot_ga_evolution

import pygad

print("="*70)
print("ACT-LSTM FULL ATTACK & VISUALIZATION")
print("="*70)

device = get_device()
power_monitor = PowerMonitor(0.01)

model = load_act(device=device)
seed_data, mean, std = load_seed_from_csv()
flat_seed = seed_data.flatten()

history = AttackHistory()


def measure_inference(input_tensor, num_reps=20):
    power_monitor.start()
    latencies = []
    ponders = []
    for _ in range(num_reps):
        start = time.perf_counter()
        with torch.no_grad():
            _, ponder = model(input_tensor)
        end = time.perf_counter()
        latencies.append(end - start)
        ponders.append(ponder.item())
    power_monitor.stop()
    stats = power_monitor.get_energy_stats()
    return {
        'latency': np.mean(latencies),
        'ponder': np.mean(ponders),
        'power': stats['avg_power']
    }


def fitness_func(ga_instance, solution, solution_idx):
    input_tensor = torch.tensor(solution.reshape(1, CONTEXT_LEN, NUM_FEATURES), dtype=torch.float32).to(device)
    stats = measure_inference(input_tensor)
    fitness = float(stats['ponder'] ** 2)
    history.record_solution(fitness, solution, ponder=stats['ponder'],
                            latency=stats['latency'], power=stats['power'])
    return fitness


def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    best = history.end_generation(gen, extra_columns={
        'best_ponder': 'ponder', 'best_latency': 'latency',
        'best_power': 'power', 'best_fitness': 'fitness'
    })
    if best:
        print(f"Gen {gen:3d}: Ponder={best['ponder']:.2f}/{MAX_PONDERS} | "
              f"Latency={best['latency']*1000:.3f}ms | Power={best['power']:.1f}W")


if __name__ == "__main__":
    print("\nStarting ACT-LSTM Full Attack...")

    print("Warming up...", end="", flush=True)
    dummy = torch.randn(1, CONTEXT_LEN, NUM_FEATURES).to(device)
    for _ in range(50):
        model(dummy)
    print(" Done.")

    base_tensor = torch.tensor(seed_data.reshape(1, CONTEXT_LEN, NUM_FEATURES), dtype=torch.float32).to(device)
    base_stats = measure_inference(base_tensor)
    print(f"Baseline: Ponder={base_stats['ponder']:.2f}, Latency={base_stats['latency']*1000:.3f}ms, Power={base_stats['power']:.1f}W")

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
    adv_tensor = torch.tensor(best_solution.reshape(1, CONTEXT_LEN, NUM_FEATURES), dtype=torch.float32).to(device)
    adv_stats = measure_inference(adv_tensor)

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Baseline Ponders: {base_stats['ponder']:.2f} | Attack Ponders: {adv_stats['ponder']:.2f}")
    print(f"Baseline Latency: {base_stats['latency']*1000:.3f}ms | Attack Latency: {adv_stats['latency']*1000:.3f}ms")
    print(f"Baseline Power: {base_stats['power']:.1f}W | Attack Power: {adv_stats['power']:.1f}W")
    slowdown = adv_stats['latency'] / base_stats['latency'] if base_stats['latency'] > 0 else 1.0
    print(f"Slowdown Factor: {slowdown:.2f}x")
    print("="*70)

    np.save("act_latency_best_input.npy", best_solution)

    plot_ga_evolution(history.generation_data,
                      {'ponder': base_stats['ponder'], 'latency': base_stats['latency'],
                       'power': base_stats['power']},
                      [
                          {'data_key': 'best_ponder', 'ylabel': 'Ponder Steps', 'title': 'Ponder Steps Evolution', 'color': 'b', 'baseline_key': 'ponder'},
                          {'data_key': 'best_latency', 'ylabel': 'Latency (ms)', 'title': 'Latency Evolution', 'scale': 1000, 'baseline_key': 'latency'},
                          {'data_key': 'best_power', 'ylabel': 'Power (W)', 'title': 'Power Evolution', 'color': 'orange', 'baseline_key': 'power'},
                          {'data_key': 'best_fitness', 'ylabel': 'Fitness', 'title': 'Fitness Evolution', 'color': 'g'},
                      ],
                      'ACT-LSTM Full Attack Results', "act_latency_results_full.png")

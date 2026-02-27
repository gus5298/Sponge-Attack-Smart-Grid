import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import argparse

from config import CONTEXT_LEN, NUM_FEATURES, POPULATION_SIZE, OUTPUT_DIR
from utils.model_loader import load_seed_from_csv, load_chronos
from utils.bitflip_oracle import BitFlipOracle
from utils.ga_operators import uniform_crossover, bitflip_mutation, create_bitflip_population
from utils.attack_runner import AttackHistory, create_ga, run_ga
from utils.visualization import plot_ga_evolution

parser = argparse.ArgumentParser()
parser.add_argument("--generations", type=int, default=30)
args = parser.parse_args()

print("=" * 70)
print("CHRONOS BIT-FLIP SPONGE ATTACK")
print("=" * 70)

device = "cpu"
OUT = os.path.join(OUTPUT_DIR, "chronos")

seed_data, mean, std = load_seed_from_csv()
flat_seed = seed_data.flatten()

print("\nLoading Chronos model for whitebox weights...")
pipeline = load_chronos(device=device)
t5_model = pipeline.model.model  # T5ForConditionalGeneration

oracle = BitFlipOracle(model=t5_model)

BASELINE_FLIPS = oracle.count_flips(seed_data)
print(f"\nBaseline Bit-Flips: {BASELINE_FLIPS:,}")

history = AttackHistory()
history.hof_max = 5


def fitness_func(ga_instance, solution, solution_idx):
    input_array = solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    flips = oracle.count_flips(input_array)
    fitness = flips / BASELINE_FLIPS if BASELINE_FLIPS > 0 else float(flips)
    history.record_solution(fitness, solution, flips=flips)
    return float(fitness)


def on_generation(ga_instance):
    gen = ga_instance.generations_completed
    history.current_gen_fitness = []
    history.current_gen_solutions = []

    all_fitness = list(ga_instance.last_generation_fitness)
    best_idx = int(np.argmax(all_fitness))
    best_sol = ga_instance.population[best_idx]
    best_input = best_sol.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    best_flips = oracle.count_flips(best_input)

    flip_pct = ((best_flips / BASELINE_FLIPS) - 1) * 100 if BASELINE_FLIPS > 0 else 0
    history.generation_data.setdefault('gen', []).append(gen)
    history.generation_data.setdefault('max_fitness', []).append(max(all_fitness))
    history.generation_data.setdefault('avg_fitness', []).append(float(np.mean(all_fitness)))
    history.generation_data.setdefault('best_flips', []).append(best_flips)
    history.generation_data.setdefault('flip_increase_pct', []).append(flip_pct)

    print(f"Gen {gen:3d}: Flips={best_flips:,} ({flip_pct:+.1f}%) | Fitness={max(all_fitness):.4f}")


initial_population = create_bitflip_population(flat_seed, POPULATION_SIZE)

ga_instance = create_ga(
    fitness_func=fitness_func, on_generation=on_generation,
    num_genes=len(flat_seed), num_generations=args.generations,
    initial_population=initial_population, gene_space=None,
    crossover_type=uniform_crossover, mutation_type=bitflip_mutation,
    keep_elitism=5,
)

if __name__ == "__main__":
    best_solution, _ = run_ga(ga_instance, "Chronos Bit-Flip Sponge Attack")

    adv_input = best_solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    adv_flips = oracle.count_flips(adv_input)
    flip_increase = ((adv_flips / BASELINE_FLIPS) - 1) * 100 if BASELINE_FLIPS > 0 else 0

    print("\n" + "=" * 70)
    print("FINAL RESULTS - CHRONOS BIT-FLIP ATTACK")
    print("=" * 70)
    print(f"Baseline Bit-Flips: {BASELINE_FLIPS:,}")
    print(f"Attack Bit-Flips:   {adv_flips:,} ({flip_increase:+.1f}%)")
    print("=" * 70)

    if flip_increase > 50:
        print("\n[SUCCESS] Significant increase in bit-flip activity achieved!")
    else:
        print("\n[LIMITED] Bit-flip increase was modest.")

    prefix = "chronos_bitflip"
    history.save(prefix, OUT)

    plot_ga_evolution(history.generation_data,
                      {'flips': BASELINE_FLIPS},
                      [
                          {'data_key': 'max_fitness', 'ylabel': 'Fitness (Flip Ratio)', 'title': 'Fitness Evolution', 'cummax': True},
                          {'data_key': 'flip_increase_pct', 'ylabel': 'Increase over Baseline (%)', 'title': 'Bit-Flip Improvement', 'color': 'b', 'cummax': True},
                      ],
                      'Chronos Bit-Flip Sponge Attack', os.path.join(OUT, f"{prefix}_results.png"))

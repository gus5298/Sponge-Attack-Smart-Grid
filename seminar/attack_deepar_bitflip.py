"""
DeepAR Bit-Flip Sponge Attack

Uses the BitFlipOracle to search for inputs that maximize
bit-level transitions, simulating hardware stress without
requiring hardware power sensors.

Modes:
    --mode blackbox: Maximize transitions within the input.
    --mode whitebox: Maximize bit differences with model weights.
"""
import numpy as np
import argparse

from config import CONTEXT_LEN, NUM_FEATURES, POPULATION_SIZE
from utils.model_loader import load_deepar, load_seed, get_device
from utils.bitflip_oracle import BitFlipOracle
from utils.ga_operators import alternating_crossover, turbulence_mutation, create_bitflip_population
from utils.attack_runner import AttackHistory, create_ga, run_ga
from utils.visualization import plot_ga_evolution

parser = argparse.ArgumentParser()
parser.add_argument("--generations", type=int, default=30)
parser.add_argument("--mode", type=str, choices=["blackbox", "whitebox"], default="blackbox")
args = parser.parse_args()

print("=" * 70)
print(f"DeepAR BIT-FLIP SPONGE ATTACK ({args.mode.upper()} MODE)")
print("=" * 70)

device = get_device()
model, checkpoint = load_deepar(device=device)
seed_data, mean, std = load_seed(checkpoint=checkpoint)
flat_seed = seed_data.flatten()

oracle = BitFlipOracle(mode=args.mode, model=model if args.mode == "whitebox" else None)

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
    best = history.end_generation(gen, extra_columns={'best_flips': 'flips'})
    if best:
        flip_change = ((best['flips'] / BASELINE_FLIPS) - 1) * 100 if BASELINE_FLIPS > 0 else 0
        print(f"Gen {gen:3d}: Flips={best['flips']:,} ({flip_change:+.1f}%) | Fitness={best['fitness']:.4f}")


initial_population = create_bitflip_population(flat_seed, POPULATION_SIZE)

ga_instance = create_ga(
    fitness_func=fitness_func, on_generation=on_generation,
    num_genes=len(flat_seed), num_generations=args.generations,
    initial_population=initial_population, gene_space=None,
    crossover_type=alternating_crossover, mutation_type=turbulence_mutation,
)

if __name__ == "__main__":
    best_solution, _ = run_ga(ga_instance, "Bit-Flip Sponge Attack")

    adv_input = best_solution.reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    adv_flips = oracle.count_flips(adv_input)
    flip_increase = ((adv_flips / BASELINE_FLIPS) - 1) * 100 if BASELINE_FLIPS > 0 else 0

    print("\n" + "=" * 70)
    print(f"FINAL RESULTS - DeepAR BIT-FLIP ATTACK ({args.mode.upper()})")
    print("=" * 70)
    print(f"Baseline Bit-Flips: {BASELINE_FLIPS:,}")
    print(f"Attack Bit-Flips:   {adv_flips:,} ({flip_increase:+.1f}%)")
    print("=" * 70)

    if flip_increase > 50:
        print("\n[SUCCESS] Significant increase in bit-flip activity achieved!")
    else:
        print("\n[LIMITED] Bit-flip increase was modest.")

    prefix = f"deepar_bitflip_{args.mode}"
    history.save(prefix)

    plot_ga_evolution(history.generation_data,
                      {'flips': BASELINE_FLIPS},
                      [
                          {'data_key': 'max_fitness', 'ylabel': 'Fitness (Flip Ratio)', 'title': 'Fitness Evolution'},
                          {'data_key': 'best_flips', 'ylabel': 'Bit-Flips', 'title': 'Bit-Flip Evolution', 'color': 'b', 'baseline_key': 'flips'},
                      ],
                      f'DeepAR Bit-Flip Sponge Attack ({args.mode.upper()})', f"{prefix}_results.png")

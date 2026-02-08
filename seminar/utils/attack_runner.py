import time
import numpy as np
import pygad

from config import (CONTEXT_LEN, NUM_FEATURES, POPULATION_SIZE,
                    NUM_PARENTS_MATING, KEEP_ELITISM)


class AttackHistory:
    def __init__(self):
        self.generation_data = {}
        self.hall_of_fame = []
        self.hof_max = 10
        self.current_gen_fitness = []
        self.current_gen_solutions = []
        self.global_best = {'fitness': 0, 'solution': None}

    def init_columns(self, columns):
        self.generation_data = {col: [] for col in columns}

    def record_solution(self, fitness, solution, **metrics):
        self.current_gen_fitness.append(fitness)
        entry = {'fitness': fitness, 'solution': solution.copy(), **metrics}
        self.current_gen_solutions.append(entry)
        if fitness > self.global_best['fitness']:
            self.global_best = entry.copy()
        self.hall_of_fame.append(entry)
        self.hall_of_fame.sort(key=lambda x: x['fitness'], reverse=True)
        while len(self.hall_of_fame) > self.hof_max:
            self.hall_of_fame.pop()
        return fitness

    def end_generation(self, gen, extra_columns=None):
        if not self.current_gen_fitness:
            return
        max_fit = max(self.current_gen_fitness)
        avg_fit = np.mean(self.current_gen_fitness)
        best = max(self.current_gen_solutions, key=lambda x: x['fitness'])
        self.generation_data.setdefault('gen', []).append(gen)
        self.generation_data.setdefault('max_fitness', []).append(max_fit)
        self.generation_data.setdefault('avg_fitness', []).append(avg_fit)
        if extra_columns:
            for col, key in extra_columns.items():
                self.generation_data.setdefault(col, []).append(best.get(key, 0))
        self.current_gen_fitness = []
        self.current_gen_solutions = []
        return best

    def save(self, prefix):
        best_sol = self.global_best.get('solution')
        if best_sol is not None:
            np.save(f"{prefix}_best_input.npy", best_sol)
        if self.generation_data:
            np.savez(f"{prefix}_generation_data.npz", **self.generation_data)
        for i, hof in enumerate(self.hall_of_fame):
            np.save(f"{prefix}_hof_{i+1}.npy", hof['solution'])


def safe_ratio(numerator, denominator):
    return numerator / denominator if denominator > 1e-9 else 1.0


def pct_change(adv_val, base_val):
    return (safe_ratio(adv_val, base_val) - 1) * 100


def print_results(title, metrics):
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS - {title}")
    print("=" * 70)
    print(f"{'Metric':<20} {'Baseline':>15} {'Adversarial':>15} {'Change':>10}")
    print("-" * 70)
    for name, base_val, adv_val, fmt in metrics:
        change = pct_change(adv_val, base_val)
        print(f"{name:<20} {fmt.format(base_val):>15} {fmt.format(adv_val):>15} {change:+.1f}%")
    print("=" * 70)


def create_ga(fitness_func, on_generation, num_genes, num_generations,
              initial_population, crossover_type, mutation_type, gene_space=None):
    return pygad.GA(
        num_generations=num_generations,
        num_parents_mating=NUM_PARENTS_MATING,
        fitness_func=fitness_func,
        sol_per_pop=POPULATION_SIZE,
        num_genes=num_genes,
        parent_selection_type="tournament",
        K_tournament=3,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        initial_population=initial_population,
        gene_space=gene_space,
        on_generation=on_generation,
        keep_elitism=KEEP_ELITISM,
        suppress_warnings=True
    )


def run_ga(ga_instance, title="Attack"):
    print(f"\nStarting {title}\n")
    start_time = time.time()
    ga_instance.run()
    total_time = time.time() - start_time
    print(f"\nTotal GA time: {total_time:.1f}s")
    best_solution, best_fitness, _ = ga_instance.best_solution()
    return best_solution, best_fitness

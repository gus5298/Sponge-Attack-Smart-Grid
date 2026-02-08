import numpy as np
from config import CONTEXT_LEN, NUM_FEATURES, MUTATION_PERCENT


def time_slice_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        p1 = parents[idx % parents.shape[0], :].reshape(CONTEXT_LEN, NUM_FEATURES)
        p2 = parents[(idx + 1) % parents.shape[0], :].reshape(CONTEXT_LEN, NUM_FEATURES)
        pt = np.random.randint(1, CONTEXT_LEN)
        child = np.vstack([p1[:pt], p2[pt:]])
        offspring.append(child.flatten())
        idx += 1
    return np.array(offspring)


def alternating_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) < offspring_size[0]:
        p1 = parents[idx % parents.shape[0], :]
        p2 = parents[(idx + 1) % parents.shape[0], :]
        child = np.where(np.arange(len(p1)) % 2 == 0, p1, p2)
        offspring.append(child)
        idx += 1
    return np.array(offspring)


DENORMALIZED_VALUES = [1e-42, 1e-43, 1e-44, 1e-45, -1e-42, -1e-43, -1e-44, -1e-45]


def latency_mutation(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.random() < MUTATION_PERCENT / 100:
                r = np.random.random()
                if r < 0.40:
                    if np.random.random() < 0.5:
                        offspring[idx, gene_idx] = np.random.uniform(1.5, 3.0)
                    else:
                        offspring[idx, gene_idx] = np.random.uniform(-3.0, -1.5)
                elif r < 0.70:
                    offspring[idx, gene_idx] = np.float32(np.random.choice(DENORMALIZED_VALUES))
                else:
                    offspring[idx, gene_idx] = np.random.uniform(-5.0, 5.0)
    return offspring


def energy_mutation(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.random() < MUTATION_PERCENT / 100:
                r = np.random.random()
                if r < 0.50:
                    if np.random.random() < 0.5:
                        offspring[idx, gene_idx] = np.random.uniform(1.5, 3.0)
                    else:
                        offspring[idx, gene_idx] = np.random.uniform(-3.0, -1.5)
                elif r < 0.75:
                    offspring[idx, gene_idx] = np.random.normal(0, 0.5)
                else:
                    offspring[idx, gene_idx] = np.random.choice([5.0, -5.0])
    return offspring


def energy_sponge_mutation(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.random() < MUTATION_PERCENT / 100:
                r = np.random.random()
                val = offspring[idx, gene_idx]
                if r < 0.50:
                    direction = np.random.choice([1, -1])
                    jump = np.random.uniform(0.1, 2.0)
                    offspring[idx, gene_idx] = val + (direction * jump)
                elif r < 0.75:
                    offspring[idx, gene_idx] = np.random.uniform(-1e4, 1e4)
                elif r < 0.92:
                    offspring[idx, gene_idx] = np.random.normal(0, 0.5)
                else:
                    offspring[idx, gene_idx] = np.float32(np.random.choice(DENORMALIZED_VALUES))
    return offspring


def turbulence_mutation(offspring, ga_instance):
    for idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.random() < MUTATION_PERCENT / 100:
                r = np.random.random()
                if r < 0.40:
                    offspring[idx, gene_idx] = -offspring[idx, gene_idx] + np.random.normal(0, 0.1)
                elif r < 0.70:
                    exponent = np.random.randint(-10, 10)
                    sign = np.random.choice([-1, 1])
                    offspring[idx, gene_idx] = sign * (2.0 ** exponent)
                elif r < 0.90:
                    offspring[idx, gene_idx] = np.random.uniform(-100, 100)
                else:
                    offspring[idx, gene_idx] = np.float32(np.random.choice([
                        1e-40, 1e-41, 1e-42, 1e-43, 1e-44, 1e-45,
                        -1e-40, -1e-41, -1e-42, -1e-43, -1e-44, -1e-45
                    ]))
    return offspring


def create_energy_population(flat_seed, pop_size, mode):
    n = len(flat_seed)
    population = []
    if mode == "constrained":
        for _ in range(pop_size):
            population.append(flat_seed + np.random.normal(0, 0.5, n))
        gene_space = {'low': -10.0, 'high': 10.0}
    else:
        n_per = pop_size // 5
        for _ in range(n_per):
            population.append(np.random.uniform(-50, 50, n))
        for _ in range(n_per):
            x = np.zeros(n)
            x[::2] = np.random.uniform(10, 100, len(x[::2]))
            x[1::2] = np.random.uniform(-100, -10, len(x[1::2]))
            population.append(x)
        for _ in range(n_per):
            population.append(np.random.normal(0, 1, n))
        for _ in range(n_per):
            population.append(np.random.uniform(-1000, 1000, n))
        for _ in range(pop_size - 4 * n_per):
            population.append(flat_seed + np.random.normal(0, 10, n))
        gene_space = None
    return np.array(population, dtype=np.float32), gene_space


def create_latency_population(flat_seed, pop_size, mode):
    n = len(flat_seed)
    population = []
    if mode == "constrained":
        for _ in range(pop_size):
            population.append(flat_seed + np.random.normal(0, 0.5, n))
        gene_space = {'low': -10.0, 'high': 10.0}
    else:
        n_per = pop_size // 5
        for _ in range(n_per):
            population.append(np.random.choice(DENORMALIZED_VALUES, n).astype(np.float32))
        for _ in range(n_per):
            population.append(np.random.uniform(-1e35, 1e35, n).astype(np.float32))
        for _ in range(n_per):
            x = np.zeros(n)
            for i in range(n):
                x[i] = 10 ** np.random.uniform(-40, 35)
                if np.random.random() < 0.5:
                    x[i] = -x[i]
            population.append(x.astype(np.float32))
        for _ in range(n_per):
            x = np.zeros(n)
            x[::2] = 1e30
            x[1::2] = 1e-40
            population.append(x.astype(np.float32))
        for _ in range(pop_size - 4 * n_per):
            population.append(flat_seed + np.random.normal(0, 10, n))
        gene_space = None
    return np.array(population, dtype=np.float32), gene_space


def create_bitflip_population(flat_seed, pop_size):
    n = len(flat_seed)
    population = []
    n_per = pop_size // 4
    for _ in range(n_per):
        x = np.random.uniform(1, 10, n)
        x[1::2] *= -1
        population.append(x)
    for _ in range(n_per):
        exponents = np.random.randint(-5, 10, n)
        signs = np.random.choice([-1, 1], n)
        population.append(signs * (2.0 ** exponents))
    for _ in range(n_per):
        population.append(np.random.randn(n) * 10)
    for _ in range(pop_size - 3 * n_per):
        population.append(flat_seed * np.random.choice([-1, 1], n))
    return np.array(population, dtype=np.float32)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import numpy as np

from config import (CONTEXT_LEN, NUM_FEATURES, PREDICTION_LEN, BASELINE_REPS,
                    VERIFICATION_REPS, PGD_EPSILON, PGD_ALPHA, PGD_NUM_STEPS,
                    PGD_WARMUP_REPS, OUTPUT_DIR)
from utils.model_loader import load_seed_from_csv, load_chronos
from utils.chronos_wrapper import ChronosWrapper
from utils.power_monitor import PowerMonitor
from utils.metrics import measure_energy
from utils.pgd import PGDAttack
from utils.attack_runner import print_results
from utils.visualization import plot_pgd_results

print("=" * 70)
print("CHRONOS PGD ENERGY ATTACK (WHITE-BOX)")
print("=" * 70)

device = "cpu"
power_monitor = PowerMonitor(0.01, cpu_only=True)
OUT = os.path.join(OUTPUT_DIR, "chronos")

print("\nLoading Chronos model...")
pipeline = load_chronos(device=device)
wrapper = ChronosWrapper(pipeline)
wrapper.train()

seed_data, mean, std = load_seed_from_csv()
seed_tensor = torch.tensor(seed_data, dtype=torch.float32).unsqueeze(0).to(device)


def make_prediction(input_array):
    if len(input_array.shape) == 2:
        univariate = input_array[:, 0]
    else:
        univariate = input_array
    inp = torch.tensor(univariate, dtype=torch.float32)
    pipeline.predict(inp, prediction_length=PREDICTION_LEN)


def energy_loss_fn(output, x):
    return output.abs().sum() + (output ** 2).sum() * 0.1


print("\nMeasuring baseline...")
baseline_stats = measure_energy(make_prediction, seed_data, power_monitor, device, num_reps=BASELINE_REPS)
print(f"Baseline Energy: {baseline_stats['energy_per_inference'] * 1000:.3f}mJ")

if __name__ == "__main__":
    print("\nStarting PGD Attack (maximizing energy consumption)...")

    dummy = torch.randn(1, CONTEXT_LEN, NUM_FEATURES).to(device)
    for _ in range(PGD_WARMUP_REPS):
        wrapper(dummy)

    attack = PGDAttack(
        model=wrapper, loss_fn=energy_loss_fn,
        epsilon=PGD_EPSILON, alpha=PGD_ALPHA, num_steps=PGD_NUM_STEPS,
        device=device, maximize=True
    )

    step_data = {'step': [], 'loss': [], 'best_loss': []}

    def on_step(step, loss, best_loss):
        step_data['step'].append(step)
        step_data['loss'].append(loss)
        step_data['best_loss'].append(best_loss)
        if step % 10 == 0:
            print(f"Step {step:3d}: Loss={loss:.2f} (Best={best_loss:.2f})")

    adv_tensor, _ = attack.attack(seed_tensor, callback=on_step)

    adv_input = adv_tensor.squeeze(0).cpu().numpy()

    adv_stats = measure_energy(make_prediction, adv_input, power_monitor, device, VERIFICATION_REPS)
    base_stats = measure_energy(make_prediction, seed_data, power_monitor, device, VERIFICATION_REPS)

    print_results("CHRONOS PGD ENERGY ATTACK", [
        ("Energy (mJ)", base_stats['energy_per_inference'] * 1000, adv_stats['energy_per_inference'] * 1000, "{:.3f}"),
        ("Power (W)", base_stats['avg_power'], adv_stats['avg_power'], "{:.1f}"),
        ("Latency (ms)", base_stats['latency'] * 1000, adv_stats['latency'] * 1000, "{:.2f}"),
    ])

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "chronos_pgd_energy_best_input.npy"), adv_input)

    plot_pgd_results(step_data, base_stats, adv_stats,
                     [('Energy (mJ)', base_stats['energy_per_inference'] * 1000, adv_stats['energy_per_inference'] * 1000),
                      ('Power (W)', base_stats['avg_power'], adv_stats['avg_power']),
                      ('Latency (ms)', base_stats['latency'] * 1000, adv_stats['latency'] * 1000)],
                     'Chronos PGD Energy Attack', os.path.join(OUT, 'chronos_pgd_energy_results.png'))

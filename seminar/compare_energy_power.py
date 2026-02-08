"""
Compare Energy and Power: DeepAR vs Chronos (with averaged measurements)
=========================================================================
Creates histograms with error bars from multiple measurement runs.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import os

from config import CONTEXT_LEN, PREDICTION_LEN, NUM_FEATURES
from utils.model_loader import load_deepar, load_seed, make_predictor, get_device
from utils.power_monitor import PowerMonitor
from utils.metrics import measure_energy

print("=" * 70)
print("ENERGY & POWER COMPARISON (AVERAGED MEASUREMENTS)")
print("=" * 70)

device = get_device()
power_monitor = PowerMonitor(sample_interval=0.005)
NUM_RUNS = 3  # Number of measurement runs for averaging

# ============ DeepAR ============
print("\n[1/4] Loading DeepAR model...")
deepar_model, checkpoint = load_deepar(device=device)
deepar_baseline, _, _ = load_seed(checkpoint=checkpoint)

if os.path.exists("deepar_bitflip_whitebox_best_input.npy"):
    deepar_adv = np.load("deepar_bitflip_whitebox_best_input.npy").reshape(CONTEXT_LEN, NUM_FEATURES)
else:
    deepar_adv = deepar_baseline

deepar_predict = make_predictor(deepar_model, device)

print(f"[2/4] Measuring DeepAR energy ({NUM_RUNS} runs x 500 reps)...")
deepar_base_energies = []
deepar_adv_energies = []
deepar_base_powers = []
deepar_adv_powers = []

for i in range(NUM_RUNS):
    base_stats = measure_energy(deepar_predict, deepar_baseline, power_monitor, device, num_reps=500)
    adv_stats = measure_energy(deepar_predict, deepar_adv, power_monitor, device, num_reps=500)
    deepar_base_energies.append(base_stats['energy_per_inference'] * 1000)
    deepar_adv_energies.append(adv_stats['energy_per_inference'] * 1000)
    deepar_base_powers.append(base_stats['avg_power'])
    deepar_adv_powers.append(adv_stats['avg_power'])
    print(f"  Run {i+1}: Base={base_stats['energy_per_inference']*1000:.2f}mJ, Adv={adv_stats['energy_per_inference']*1000:.2f}mJ")

# ============ Chronos ============
print("\n[3/4] Loading Chronos model...")
from chronos import ChronosPipeline

chronos_pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-base",
    device_map=device,
    torch_dtype=torch.float32,
)

chronos_baseline, _, _ = load_seed_data(DATA_PATH, CONTEXT_LEN)

if os.path.exists("chronos_latency_best_input.npy"):
    chronos_adv = np.load("chronos_latency_best_input.npy").reshape(CONTEXT_LEN, NUM_FEATURES)
else:
    chronos_adv = chronos_baseline

def chronos_predict(input_array):
    inp = torch.tensor(input_array[:, 0], dtype=torch.float32)
    chronos_pipeline.predict(inp, prediction_length=PREDICTION_LEN)

print(f"[4/4] Measuring Chronos energy ({NUM_RUNS} runs x 5 reps)...")
chronos_base_energies = []
chronos_adv_energies = []
chronos_base_powers = []
chronos_adv_powers = []

for i in range(NUM_RUNS):
    base_stats = measure_energy(chronos_predict, chronos_baseline, power_monitor, device, num_reps=5)
    adv_stats = measure_energy(chronos_predict, chronos_adv, power_monitor, device, num_reps=5)
    chronos_base_energies.append(base_stats['energy_per_inference'] * 1000)
    chronos_adv_energies.append(adv_stats['energy_per_inference'] * 1000)
    chronos_base_powers.append(base_stats['avg_power'])
    chronos_adv_powers.append(adv_stats['avg_power'])
    print(f"  Run {i+1}: Base={base_stats['energy_per_inference']*1000:.0f}mJ, Adv={adv_stats['energy_per_inference']*1000:.0f}mJ")

# ============ Create Comparison Plot with Error Bars ============
print("\nGenerating comparison histogram with error bars...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Calculate means and stds
models = ['DeepAR\nBaseline', 'DeepAR\nAdversarial', 'Chronos\nBaseline', 'Chronos\nAdversarial']

energy_means = [
    np.mean(deepar_base_energies),
    np.mean(deepar_adv_energies),
    np.mean(chronos_base_energies),
    np.mean(chronos_adv_energies)
]
energy_stds = [
    np.std(deepar_base_energies),
    np.std(deepar_adv_energies),
    np.std(chronos_base_energies),
    np.std(chronos_adv_energies)
]

power_means = [
    np.mean(deepar_base_powers),
    np.mean(deepar_adv_powers),
    np.mean(chronos_base_powers),
    np.mean(chronos_adv_powers)
]
power_stds = [
    np.std(deepar_base_powers),
    np.std(deepar_adv_powers),
    np.std(chronos_base_powers),
    np.std(chronos_adv_powers)
]

colors = ['#4c72b0', '#dd8452', '#4c72b0', '#dd8452']
hatches = ['', '', '//', '//']

# Energy subplot
x = np.arange(len(models))
bars1 = axes[0].bar(x, energy_means, yerr=energy_stds, capsize=5, 
                     color=colors, edgecolor='black', linewidth=1.2)
for bar, hatch in zip(bars1, hatches):
    bar.set_hatch(hatch)
axes[0].set_ylabel('Energy per Inference (mJ)', fontsize=12)
axes[0].set_title('Energy Consumption (Mean ± Std)', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val, std in zip(bars1, energy_means, energy_stds):
    height = bar.get_height()
    axes[0].annotate(f'{val:.1f}±{std:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.5),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

# Power subplot
bars2 = axes[1].bar(x, power_means, yerr=power_stds, capsize=5,
                     color=colors, edgecolor='black', linewidth=1.2)
for bar, hatch in zip(bars2, hatches):
    bar.set_hatch(hatch)
axes[1].set_ylabel('Average Power (W)', fontsize=12)
axes[1].set_title('Power Consumption (Mean ± Std)', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val, std in zip(bars2, power_means, power_stds):
    height = bar.get_height()
    axes[1].annotate(f'{val:.1f}±{std:.1f}',
                     xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.3),
                     ha='center', va='bottom', fontsize=9, fontweight='bold')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4c72b0', edgecolor='black', label='Baseline'),
    Patch(facecolor='#dd8452', edgecolor='black', label='Adversarial'),
    Patch(facecolor='white', edgecolor='black', hatch='//', label='Chronos (Transformer)')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=11, 
           bbox_to_anchor=(0.5, 1.02))

plt.suptitle('Sponge Attack: Energy & Power Comparison (Averaged)', fontsize=16, fontweight='bold', y=1.08)
plt.tight_layout()
plt.savefig('energy_power_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: energy_power_comparison.png")

# Print summary
print("\n" + "=" * 70)
print("SUMMARY (Mean ± Std)")
print("=" * 70)
print(f"{'Model':<15} {'Metric':<12} {'Baseline':<18} {'Adversarial':<18} {'Change':<10}")
print("-" * 70)

deepar_energy_change = ((np.mean(deepar_adv_energies) / np.mean(deepar_base_energies)) - 1) * 100
chronos_energy_change = ((np.mean(chronos_adv_energies) / np.mean(chronos_base_energies)) - 1) * 100

print(f"{'DeepAR':<15} {'Energy(mJ)':<12} {np.mean(deepar_base_energies):.2f}±{np.std(deepar_base_energies):.2f}      {np.mean(deepar_adv_energies):.2f}±{np.std(deepar_adv_energies):.2f}      {deepar_energy_change:+.1f}%")
print(f"{'DeepAR':<15} {'Power(W)':<12} {np.mean(deepar_base_powers):.2f}±{np.std(deepar_base_powers):.2f}      {np.mean(deepar_adv_powers):.2f}±{np.std(deepar_adv_powers):.2f}      {((np.mean(deepar_adv_powers)/np.mean(deepar_base_powers))-1)*100:+.1f}%")
print(f"{'Chronos':<15} {'Energy(mJ)':<12} {np.mean(chronos_base_energies):.0f}±{np.std(chronos_base_energies):.0f}      {np.mean(chronos_adv_energies):.0f}±{np.std(chronos_adv_energies):.0f}      {chronos_energy_change:+.1f}%")
print(f"{'Chronos':<15} {'Power(W)':<12} {np.mean(chronos_base_powers):.2f}±{np.std(chronos_base_powers):.2f}      {np.mean(chronos_adv_powers):.2f}±{np.std(chronos_adv_powers):.2f}      {((np.mean(chronos_adv_powers)/np.mean(chronos_base_powers))-1)*100:+.1f}%")
print("=" * 70)

# Significance check
print("\n[CONCLUSION]")
deepar_std = max(np.std(deepar_base_energies), np.std(deepar_adv_energies))
chronos_std = max(np.std(chronos_base_energies), np.std(chronos_adv_energies))
deepar_diff = abs(np.mean(deepar_adv_energies) - np.mean(deepar_base_energies))
chronos_diff = abs(np.mean(chronos_adv_energies) - np.mean(chronos_base_energies))

if deepar_diff < deepar_std:
    print("DeepAR: ❌ No significant effect (difference within noise)")
else:
    print(f"DeepAR: Energy difference {deepar_diff:.2f}mJ > std {deepar_std:.2f}mJ")

if chronos_diff < chronos_std:
    print("Chronos: ❌ No significant effect (difference within noise)")
else:
    print(f"Chronos: Energy difference {chronos_diff:.0f}mJ > std {chronos_std:.0f}mJ")

"""
Analyze Adversarial Input Patterns for Energy Sponge Attack
============================================================
Compare high-power adversarial inputs vs baseline to understand
what patterns cause increased GPU power consumption.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from config import CONTEXT_LEN, NUM_FEATURES, ALL_FEATURES, OUTPUT_DIR
from utils.model_loader import load_seed_from_csv

print("="*70)
print("ADVERSARIAL INPUT PATTERN ANALYSIS")
print("="*70)

comparisons_dir = os.path.join(OUTPUT_DIR, 'comparisons')
os.makedirs(comparisons_dir, exist_ok=True)

baseline, _, _ = load_seed_from_csv()
print(f"Loaded baseline: {baseline.shape}")

hof_inputs = []
hof_powers = [74.2, 73.9, 75.7, 74.7, 74.6, 75.2, 74.4, 72.6, 75.0, 74.6]

for i in range(1, 11):
    try:
        hof_path = os.path.join(OUTPUT_DIR, 'chronos', f'chronos_energy_hof_{i}.npy')
        inp = np.load(hof_path).reshape(CONTEXT_LEN, NUM_FEATURES)
        hof_inputs.append(inp)
    except FileNotFoundError:
        pass

print(f"Loaded {len(hof_inputs)} Hall of Fame inputs")

if not hof_inputs:
    print("No HoF inputs found!")
    exit(1)

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("STATISTICAL COMPARISON: Baseline vs Adversarial")
print("="*70)

adversarial = np.stack(hof_inputs)
adv_mean = adversarial.mean(axis=0)

print(f"\n{'Metric':<30} {'Baseline':>15} {'Adversarial':>15} {'Ratio':>10}")
print("-"*70)

base_flat = baseline.flatten()
adv_flat = adv_mean.flatten()

metrics = [
    ("Mean", np.mean(base_flat), np.mean(adv_flat)),
    ("Std Dev", np.std(base_flat), np.std(adv_flat)),
    ("Min", np.min(base_flat), np.min(adv_flat)),
    ("Max", np.max(base_flat), np.max(adv_flat)),
    ("Abs Mean", np.mean(np.abs(base_flat)), np.mean(np.abs(adv_flat))),
    ("Range", np.ptp(base_flat), np.ptp(adv_flat)),
    ("Variance", np.var(base_flat), np.var(adv_flat)),
]

for name, base_val, adv_val in metrics:
    ratio = adv_val / base_val if abs(base_val) > 1e-6 else float('inf')
    print(f"{name:<30} {base_val:>15.4f} {adv_val:>15.4f} {ratio:>10.2f}x")

# Per-feature analysis
print("\n" + "="*70)
print("PER-FEATURE ANALYSIS")
print("="*70)
print(f"\n{'Feature':<25} {'Base Std':>12} {'Adv Std':>12} {'Std Ratio':>12} {'Base Range':>12} {'Adv Range':>12}")
print("-"*85)

for i, feat in enumerate(ALL_FEATURES):
    base_feat = baseline[:, i]
    adv_feat = adv_mean[:, i]
    base_std = np.std(base_feat)
    adv_std = np.std(adv_feat)
    std_ratio = adv_std / base_std if base_std > 1e-6 else float('inf')
    print(f"{feat:<25} {base_std:>12.4f} {adv_std:>12.4f} {std_ratio:>12.2f}x {np.ptp(base_feat):>12.4f} {np.ptp(adv_feat):>12.4f}")

# =============================================================================
# PATTERN ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("PATTERN ANALYSIS")
print("="*70)

base_grad = np.diff(baseline, axis=0)
adv_grad = np.diff(adv_mean, axis=0)

print(f"\n1. TEMPORAL GRADIENTS (changes between timesteps):")
print(f"   Baseline avg gradient magnitude: {np.mean(np.abs(base_grad)):.4f}")
print(f"   Adversarial avg gradient magnitude: {np.mean(np.abs(adv_grad)):.4f}")
print(f"   Ratio: {np.mean(np.abs(adv_grad)) / np.mean(np.abs(base_grad)):.2f}x")

base_sign_changes = np.sum(np.diff(np.sign(baseline), axis=0) != 0)
adv_sign_changes = np.sum(np.diff(np.sign(adv_mean), axis=0) != 0)

print(f"\n2. SIGN CHANGES (oscillations):")
print(f"   Baseline sign changes: {base_sign_changes}")
print(f"   Adversarial sign changes: {adv_sign_changes}")
print(f"   Ratio: {adv_sign_changes / base_sign_changes if base_sign_changes > 0 else 0:.2f}x")


def count_unique_bins(data, n_bins=4096):
    flat = data.flatten()
    normalized = (flat - flat.min()) / (flat.max() - flat.min() + 1e-6)
    bins = (normalized * (n_bins - 1)).astype(int)
    return len(np.unique(bins))


print(f"\n3. VALUE DISTRIBUTION:")
base_bins = count_unique_bins(baseline)
adv_bins = count_unique_bins(adv_mean)
print(f"   Baseline unique bins: {base_bins}")
print(f"   Adversarial unique bins: {adv_bins}")
print(f"   Token diversity: {adv_bins / base_bins:.2f}x")

base_extreme = np.sum(np.abs(base_flat) > 3)
adv_extreme = np.sum(np.abs(adv_flat) > 3)

print(f"\n4. EXTREME VALUES (|x| > 3):")
print(f"   Baseline extreme values: {base_extreme} ({100*base_extreme/len(base_flat):.1f}%)")
print(f"   Adversarial extreme values: {adv_extreme} ({100*adv_extreme/len(adv_flat):.1f}%)")

base_corr = np.corrcoef(baseline.T)
adv_corr = np.corrcoef(adv_mean.T)

print(f"\n5. FEATURE CORRELATIONS:")
print(f"   Baseline avg abs correlation: {np.mean(np.abs(np.triu(base_corr, 1))):.4f}")
print(f"   Adversarial avg abs correlation: {np.mean(np.abs(np.triu(adv_corr, 1))):.4f}")

# =============================================================================
# KEY INSIGHTS
# =============================================================================
print("\n" + "="*70)
print("KEY INSIGHTS - What Causes Higher GPU Power?")
print("="*70)

insights = []
var_ratio = np.var(adv_flat) / np.var(base_flat)
if var_ratio > 2:
    insights.append(f"VARIANCE: Adversarial inputs have {var_ratio:.1f}x higher variance")
grad_ratio = np.mean(np.abs(adv_grad)) / np.mean(np.abs(base_grad))
if grad_ratio > 2:
    insights.append(f"GRADIENTS: {grad_ratio:.1f}x larger temporal changes")
if adv_extreme > base_extreme * 2:
    insights.append(f"EXTREMES: {adv_extreme}x more extreme values")
mag_ratio = np.mean(np.abs(adv_flat)) / np.mean(np.abs(base_flat))
if mag_ratio > 1.5:
    insights.append(f"MAGNITUDE: {mag_ratio:.1f}x larger absolute values")

for i, insight in enumerate(insights, 1):
    print(f"  {i}. {insight}")

if not insights:
    print("  No clear pattern identified - attack may be exploiting subtle embedding space properties")

# =============================================================================
# VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

axes[0, 0].plot(baseline[:, 0], 'b-', linewidth=2, label='Baseline', alpha=0.7)
axes[0, 0].plot(adv_mean[:, 0], 'r-', linewidth=2, label='Adversarial (mean)')
axes[0, 0].set_xlabel('Timestep')
axes[0, 0].set_ylabel('Power Feature')
axes[0, 0].set_title('Time Series: Baseline vs Adversarial')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(base_flat, bins=50, alpha=0.5, label='Baseline', density=True)
axes[0, 1].hist(adv_flat, bins=50, alpha=0.5, label='Adversarial', density=True)
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Value Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].bar(['Baseline', 'Adversarial'],
        [np.mean(np.abs(base_grad)), np.mean(np.abs(adv_grad))],
        color=['blue', 'red'], alpha=0.7)
axes[1, 0].set_ylabel('Mean Gradient Magnitude')
axes[1, 0].set_title('Temporal Change Rate')
axes[1, 0].grid(True, alpha=0.3)

base_vars = np.var(baseline, axis=0)
adv_vars = np.var(adv_mean, axis=0)
x = np.arange(len(ALL_FEATURES))
width = 0.35
axes[1, 1].bar(x - width/2, base_vars, width, label='Baseline', alpha=0.7)
axes[1, 1].bar(x + width/2, adv_vars, width, label='Adversarial', alpha=0.7)
axes[1, 1].set_xlabel('Feature')
axes[1, 1].set_ylabel('Variance')
axes[1, 1].set_title('Per-Feature Variance')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels([f.split('_')[0][:6] for f in ALL_FEATURES], rotation=45)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

diff = adv_mean - baseline
im = axes[2, 0].imshow(diff.T, aspect='auto', cmap='RdBu_r',
                         vmin=-np.percentile(np.abs(diff), 95), vmax=np.percentile(np.abs(diff), 95))
axes[2, 0].set_xlabel('Timestep')
axes[2, 0].set_ylabel('Feature')
axes[2, 0].set_yticks(range(len(ALL_FEATURES)))
axes[2, 0].set_yticklabels([f.split('_')[0][:8] for f in ALL_FEATURES])
axes[2, 0].set_title('Adversarial - Baseline Difference')
plt.colorbar(im, ax=axes[2, 0])

for i, inp in enumerate(hof_inputs[:3]):
    alpha = 1.0 - i * 0.25
    axes[2, 1].plot(inp[:, 0], alpha=alpha, linewidth=1.5, label=f'HoF #{i+1} ({hof_powers[i]:.1f}W)')
axes[2, 1].plot(baseline[:, 0], 'k--', linewidth=2, label='Baseline')
axes[2, 1].set_xlabel('Timestep')
axes[2, 1].set_ylabel('Power Feature')
axes[2, 1].set_title('Top 3 High-Power Inputs vs Baseline')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.suptitle('Energy Sponge Attack - Adversarial Pattern Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
save_path = os.path.join(comparisons_dir, 'adversarial_pattern_analysis.png')
plt.savefig(save_path, dpi=150)
print(f"\nSaved: {save_path}")

print("\n" + "="*70)
print("Analysis Complete")
print("="*70)

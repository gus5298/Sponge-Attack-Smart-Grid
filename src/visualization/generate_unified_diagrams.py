"""
Generate Unified Adversarial Pattern Analysis Diagrams
=======================================================
Creates comprehensive analysis plots for Chronos, DeepAR, and ACT-LSTM attacks.
Mirrors the style from analyze_adversarial_patterns.py but generalized for any model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse

from config import CONTEXT_LEN, NUM_FEATURES, FEATURE_NAMES, OUTPUT_DIR
from utils.model_loader import load_seed_from_csv
from utils.xai_helpers import load_adversarial_inputs


def compute_statistics(baseline, adversarial):
    """Compute statistical comparisons."""
    base_flat = baseline.flatten()
    adv_flat = adversarial.flatten()

    stats_dict = {
        'Mean': (np.mean(base_flat), np.mean(adv_flat)),
        'Std Dev': (np.std(base_flat), np.std(adv_flat)),
        'Min': (np.min(base_flat), np.min(adv_flat)),
        'Max': (np.max(base_flat), np.max(adv_flat)),
        'Abs Mean': (np.mean(np.abs(base_flat)), np.mean(np.abs(adv_flat))),
        'Range': (np.ptp(base_flat), np.ptp(adv_flat)),
        'Variance': (np.var(base_flat), np.var(adv_flat)),
    }

    return stats_dict


def compute_pattern_metrics(baseline, adversarial):
    """Compute pattern-based metrics."""
    # Temporal gradients
    base_grad = np.diff(baseline, axis=0)
    adv_grad = np.diff(adversarial, axis=0)

    # Sign changes (oscillations)
    base_sign_changes = np.sum(np.diff(np.sign(baseline), axis=0) != 0)
    adv_sign_changes = np.sum(np.diff(np.sign(adversarial), axis=0) != 0)

    # Extreme values
    base_extreme = np.sum(np.abs(baseline.flatten()) > 3)
    adv_extreme = np.sum(np.abs(adversarial.flatten()) > 3)

    return {
        'grad_base': np.mean(np.abs(base_grad)),
        'grad_adv': np.mean(np.abs(adv_grad)),
        'sign_changes_base': base_sign_changes,
        'sign_changes_adv': adv_sign_changes,
        'extreme_base': base_extreme,
        'extreme_adv': adv_extreme
    }


def plot_adversarial_analysis(model_name, baseline, adversarial, save_prefix, attack_type=''):
    """
    Create comprehensive 6-subplot adversarial pattern analysis.

    Args:
        model_name: Name of the model (e.g., 'Chronos', 'DeepAR')
        baseline: Baseline input array (CONTEXT_LEN, NUM_FEATURES)
        adversarial: Adversarial input array (CONTEXT_LEN, NUM_FEATURES)
        save_prefix: Prefix for saved files
        attack_type: Type of attack (e.g., 'Latency', 'Energy', 'Bit-Flip')
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 1. Time series comparison (first feature - Power)
    ax1 = axes[0, 0]
    ax1.plot(baseline[:, 0], 'b-', linewidth=2, label='Baseline', alpha=0.7)
    ax1.plot(adversarial[:, 0], 'r-', linewidth=2, label='Adversarial')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Power Feature')
    ax1.set_title('Time Series: Baseline vs Adversarial')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Value distribution histogram
    ax2 = axes[0, 1]
    base_flat = baseline.flatten()
    adv_flat = adversarial.flatten()
    ax2.hist(base_flat, bins=50, alpha=0.5, label='Baseline', density=True, color='blue')
    ax2.hist(adv_flat, bins=50, alpha=0.5, label='Adversarial', density=True, color='red')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Density')
    ax2.set_title('Value Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Gradient magnitude comparison
    ax3 = axes[1, 0]
    base_grad = np.diff(baseline, axis=0)
    adv_grad = np.diff(adversarial, axis=0)
    ax3.bar(['Baseline', 'Adversarial'],
            [np.mean(np.abs(base_grad)), np.mean(np.abs(adv_grad))],
            color=['blue', 'red'], alpha=0.7)
    ax3.set_ylabel('Mean Gradient Magnitude')
    ax3.set_title('Temporal Change Rate')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Per-feature variance comparison
    ax4 = axes[1, 1]
    base_vars = np.var(baseline, axis=0)
    adv_vars = np.var(adversarial, axis=0)
    x = np.arange(len(FEATURE_NAMES))
    width = 0.35
    ax4.bar(x - width/2, base_vars, width, label='Baseline', alpha=0.7, color='blue')
    ax4.bar(x + width/2, adv_vars, width, label='Adversarial', alpha=0.7, color='red')
    ax4.set_xlabel('Feature')
    ax4.set_ylabel('Variance')
    ax4.set_title('Per-Feature Variance')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f[:6] for f in FEATURE_NAMES], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Heatmap: Adversarial - Baseline difference
    ax5 = axes[2, 0]
    diff = adversarial - baseline
    vmax = np.percentile(np.abs(diff), 95)
    im = ax5.imshow(diff.T, aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Feature')
    ax5.set_yticks(range(len(FEATURE_NAMES)))
    ax5.set_yticklabels([f[:8] for f in FEATURE_NAMES])
    ax5.set_title('Adversarial - Baseline Difference')
    plt.colorbar(im, ax=ax5)

    # 6. Adversarial input heatmap
    ax6 = axes[2, 1]
    im2 = ax6.imshow(adversarial.T, aspect='auto', cmap='viridis')
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Feature')
    ax6.set_yticks(range(len(FEATURE_NAMES)))
    ax6.set_yticklabels([f[:8] for f in FEATURE_NAMES])
    ax6.set_title('Adversarial Input Heatmap')
    plt.colorbar(im2, ax=ax6)

    # Overall title
    title = f'{model_name} {attack_type} Attack - Pattern Analysis'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    comparisons_dir = os.path.join(OUTPUT_DIR, 'comparisons')
    os.makedirs(comparisons_dir, exist_ok=True)
    output_path = os.path.join(comparisons_dir, f'{save_prefix}_pattern_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f'Saved: {output_path}')
    plt.close()


def print_statistical_summary(model_name, baseline, adversarial, attack_type=''):
    """Print statistical comparison to console."""
    print("\n" + "=" * 70)
    print(f"{model_name} {attack_type} Attack - Statistical Analysis")
    print("=" * 70)

    stats_dict = compute_statistics(baseline, adversarial)

    print(f"\n{'Metric':<30} {'Baseline':>15} {'Adversarial':>15} {'Ratio':>10}")
    print("-" * 70)

    for name, (base_val, adv_val) in stats_dict.items():
        ratio = adv_val / base_val if abs(base_val) > 1e-6 else float('inf')
        print(f"{name:<30} {base_val:>15.4f} {adv_val:>15.4f} {ratio:>10.2f}x")

    # Pattern metrics
    patterns = compute_pattern_metrics(baseline, adversarial)

    print("\n" + "=" * 70)
    print("Pattern Analysis")
    print("=" * 70)

    grad_ratio = patterns['grad_adv'] / patterns['grad_base'] if patterns['grad_base'] > 0 else 0
    print(f"Temporal Gradients: {patterns['grad_base']:.4f} -> {patterns['grad_adv']:.4f} ({grad_ratio:.2f}x)")

    sign_ratio = patterns['sign_changes_adv'] / patterns['sign_changes_base'] if patterns['sign_changes_base'] > 0 else 0
    print(f"Sign Changes: {patterns['sign_changes_base']} -> {patterns['sign_changes_adv']} ({sign_ratio:.2f}x)")

    print(f"Extreme Values: {patterns['extreme_base']} -> {patterns['extreme_adv']}")
    print()


def discover_inputs():
    """Scan outputs/ for all *_best_input.npy files and return (model, attack, path) tuples."""
    found = []
    for model_dir in ('deepar', 'act', 'chronos'):
        scan_dir = os.path.join(OUTPUT_DIR, model_dir)
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if fname.endswith('_best_input.npy'):
                # e.g. "deepar_latency_best_input.npy" -> attack = "latency"
                prefix = fname.replace('_best_input.npy', '')
                attack = prefix.replace(f'{model_dir}_', '', 1)
                found.append((model_dir, attack, os.path.join(scan_dir, fname)))
    return found


def run_single(model, attack, adv_input_path, baseline_csv='data/Location1.csv', output_prefix=None):
    """Run analysis for a single model/attack combination."""
    if output_prefix is None:
        output_prefix = f"{model}_{attack}"

    print("=" * 70)
    print(f"UNIFIED DIAGRAM GENERATION: {model.upper()} - {attack.upper()}")
    print("=" * 70)

    baseline, _, _ = load_seed_from_csv(baseline_csv)
    adv_map = load_adversarial_inputs({'adv': adv_input_path})
    if 'adv' not in adv_map:
        print(f"WARNING: Adversarial input not found: {adv_input_path}, skipping.")
        return
    adversarial = adv_map['adv']

    print_statistical_summary(model.upper(), baseline, adversarial, attack.capitalize())
    plot_adversarial_analysis(model.upper(), baseline, adversarial, output_prefix, attack.capitalize())


def main():
    parser = argparse.ArgumentParser(description='Generate unified adversarial pattern analysis diagrams')
    parser.add_argument('--model', type=str, default=None, help='Model name (e.g., chronos, deepar, act)')
    parser.add_argument('--attack', type=str, default=None, help='Attack type (e.g., latency, energy, bitflip)')
    parser.add_argument('--adv-input', type=str, default=None, help='Path to adversarial input .npy file')
    parser.add_argument('--baseline-csv', type=str, default='data/Location1.csv', help='Path to baseline CSV')
    parser.add_argument('--output-prefix', type=str, default=None, help='Output file prefix')

    args = parser.parse_args()

    if args.model and args.attack and args.adv_input:
        # Explicit single-run mode
        run_single(args.model, args.attack, args.adv_input, args.baseline_csv, args.output_prefix)
    else:
        # Auto-discovery mode
        inputs = discover_inputs()
        if not inputs:
            print("No adversarial inputs found in outputs/. Nothing to do.")
            return
        print(f"Auto-discovery: found {len(inputs)} adversarial inputs.\n")
        for model, attack, path in inputs:
            run_single(model, attack, path, args.baseline_csv)

    print("\n" + "=" * 70)
    print("DIAGRAM GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

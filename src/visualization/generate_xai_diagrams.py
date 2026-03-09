"""
Generate XAI Diagrams for Chronos and DeepAR
============================================
Creates feature importance and baseline vs adversarial comparison diagrams
similar to the ACT-LSTM XAI analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import argparse

from config import CONTEXT_LEN, NUM_FEATURES, FEATURE_NAMES, OUTPUT_DIR
from utils.model_loader import load_seed_from_csv
from utils.xai_helpers import load_adversarial_inputs


def compute_feature_importance(baseline, adversarial):
    """
    Compute feature importance based on perturbation magnitude
    and temporal variance.
    """
    # Method 1: Absolute perturbation magnitude
    perturbation = adversarial - baseline
    importance_magnitude = np.abs(perturbation).sum(axis=0)

    # Method 2: Variance contribution
    importance_variance = np.var(perturbation, axis=0)

    # Method 3: Temporal gradient
    grad_perturbation = np.abs(np.diff(perturbation, axis=0)).sum(axis=0)

    # Normalize and combine
    importance_magnitude = importance_magnitude / importance_magnitude.max()
    importance_variance = importance_variance / (importance_variance.max() + 1e-8)
    grad_perturbation = grad_perturbation / (grad_perturbation.max() + 1e-8)

    # Combined importance score
    combined_importance = (
        0.5 * importance_magnitude +
        0.3 * importance_variance +
        0.2 * grad_perturbation
    )

    return combined_importance, importance_magnitude, importance_variance, grad_perturbation


def plot_feature_importance(importance, feature_names, model_name, attack_type, save_path):
    """Plot feature importance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    sorted_idx = np.argsort(importance)[::-1]
    y_pos = np.arange(len(feature_names))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))

    ax.barh(y_pos, importance[sorted_idx], color=colors[sorted_idx], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'{model_name} {attack_type} Attack - Feature Importance',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


def plot_perturbation_heatmap(baseline, adversarial, feature_names, model_name, attack_type, save_path):
    """Plot perturbation heatmap showing adversarial - baseline difference."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    perturbation = adversarial - baseline

    # Baseline heatmap
    im1 = axes[0].imshow(baseline.T, aspect='auto', cmap='viridis')
    axes[0].set_xlabel('Time Step', fontsize=11)
    axes[0].set_ylabel('Feature', fontsize=11)
    axes[0].set_yticks(range(len(feature_names)))
    axes[0].set_yticklabels(feature_names, fontsize=9)
    axes[0].set_title('Baseline Input', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='Value')

    # Adversarial heatmap
    im2 = axes[1].imshow(adversarial.T, aspect='auto', cmap='viridis')
    axes[1].set_xlabel('Time Step', fontsize=11)
    axes[1].set_ylabel('Feature', fontsize=11)
    axes[1].set_yticks(range(len(feature_names)))
    axes[1].set_yticklabels(feature_names, fontsize=9)
    axes[1].set_title('Adversarial Input', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Value')

    # Perturbation (difference) heatmap
    vmax = np.percentile(np.abs(perturbation), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im3 = axes[2].imshow(perturbation.T, aspect='auto', cmap='RdBu_r', norm=norm)
    axes[2].set_xlabel('Time Step', fontsize=11)
    axes[2].set_ylabel('Feature', fontsize=11)
    axes[2].set_yticks(range(len(feature_names)))
    axes[2].set_yticklabels(feature_names, fontsize=9)
    axes[2].set_title('Perturbation (Delta)', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], label='Perturbation')

    plt.suptitle(f'{model_name} {attack_type} Attack - Input Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


def plot_temporal_importance(baseline, adversarial, feature_names, model_name, attack_type, save_path):
    """Plot temporal importance showing which timesteps are most perturbed."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    perturbation = adversarial - baseline

    # Temporal perturbation magnitude (sum across features)
    temporal_mag = np.abs(perturbation).sum(axis=1)

    # Per-feature temporal plot
    for i, feat_name in enumerate(feature_names):
        axes[0].plot(np.abs(perturbation[:, i]), label=feat_name, alpha=0.7, linewidth=1.5)

    axes[0].set_xlabel('Time Step', fontsize=11)
    axes[0].set_ylabel('|Perturbation|', fontsize=11)
    axes[0].set_title('Per-Feature Perturbation Over Time', fontsize=12, fontweight='bold')
    axes[0].legend(ncol=3, fontsize=8, loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Overall temporal magnitude
    axes[1].bar(range(len(temporal_mag)), temporal_mag, color='steelblue', alpha=0.7)
    axes[1].set_xlabel('Time Step', fontsize=11)
    axes[1].set_ylabel('Total Perturbation Magnitude', fontsize=11)
    axes[1].set_title('Temporal Perturbation Intensity', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'{model_name} {attack_type} Attack - Temporal Analysis',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


def plot_comprehensive_xai(baseline, adversarial, feature_names, model_name, attack_type, save_path):
    """Create comprehensive 4-panel XAI analysis."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    perturbation = adversarial - baseline

    # 1. Feature importance
    ax1 = fig.add_subplot(gs[0, 0])
    combined_importance, _, _, _ = compute_feature_importance(baseline, adversarial)
    sorted_idx = np.argsort(combined_importance)[::-1]
    y_pos = np.arange(len(feature_names))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))
    ax1.barh(y_pos, combined_importance[sorted_idx], color=colors[sorted_idx], alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=9)
    ax1.set_xlabel('Importance Score', fontsize=10)
    ax1.set_title('Feature Importance', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # 2. Perturbation heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    vmax = np.percentile(np.abs(perturbation), 95)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im2 = ax2.imshow(perturbation.T, aspect='auto', cmap='RdBu_r', norm=norm)
    ax2.set_xlabel('Time Step', fontsize=10)
    ax2.set_ylabel('Feature', fontsize=10)
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels([f[:8] for f in feature_names], fontsize=8)
    ax2.set_title('Perturbation Heatmap', fontsize=11, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Delta')

    # 3. Temporal magnitude
    ax3 = fig.add_subplot(gs[1, :])
    temporal_mag = np.abs(perturbation).sum(axis=1)
    ax3.bar(range(len(temporal_mag)), temporal_mag, color='steelblue', alpha=0.7)
    ax3.set_xlabel('Time Step', fontsize=10)
    ax3.set_ylabel('Total Perturbation', fontsize=10)
    ax3.set_title('Temporal Perturbation Intensity', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Baseline vs Adversarial comparison (first feature)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(baseline[:, 0], 'b-', linewidth=2, label='Baseline', alpha=0.7)
    ax4.plot(adversarial[:, 0], 'r-', linewidth=2, label='Adversarial')
    ax4.set_xlabel('Time Step', fontsize=10)
    ax4.set_ylabel(f'{feature_names[0]}', fontsize=10)
    ax4.set_title('Time Series Comparison (Primary Feature)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 5. Statistical comparison
    ax5 = fig.add_subplot(gs[2, 1])
    stats_names = ['Mean', 'Std', 'Max\n|Val|', 'Range']
    base_stats = [
        np.abs(baseline).mean(),
        baseline.std(),
        np.abs(baseline).max(),
        np.ptp(baseline)
    ]
    adv_stats = [
        np.abs(adversarial).mean(),
        adversarial.std(),
        np.abs(adversarial).max(),
        np.ptp(adversarial)
    ]
    x = np.arange(len(stats_names))
    width = 0.35
    ax5.bar(x - width/2, base_stats, width, label='Baseline', alpha=0.7, color='blue')
    ax5.bar(x + width/2, adv_stats, width, label='Adversarial', alpha=0.7, color='red')
    ax5.set_ylabel('Value', fontsize=10)
    ax5.set_title('Statistical Comparison', fontsize=11, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(stats_names, fontsize=9)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'{model_name} {attack_type} Attack - XAI Analysis',
                 fontsize=15, fontweight='bold')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {save_path}')
    plt.close()


def discover_inputs():
    """Scan outputs/ for all *_best_input.npy files and return (model, attack, path) tuples."""
    found = []
    for model_dir in ('deepar', 'act', 'chronos'):
        scan_dir = os.path.join(OUTPUT_DIR, model_dir)
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if fname.endswith('_best_input.npy'):
                prefix = fname.replace('_best_input.npy', '')
                attack = prefix.replace(f'{model_dir}_', '', 1)
                found.append((model_dir, attack, os.path.join(scan_dir, fname)))
    return found


def run_single(model, attack, adv_input_path, baseline_csv='data/Location1.csv', output_prefix=None):
    """Run XAI analysis for a single model/attack combination."""
    if output_prefix is None:
        output_prefix = f"{model}_{attack}"

    print("=" * 70)
    print(f"XAI DIAGRAM GENERATION: {model.upper()} - {attack.upper()}")
    print("=" * 70)

    baseline, _, _ = load_seed_from_csv(baseline_csv)
    adv_map = load_adversarial_inputs({'adv': adv_input_path})
    if 'adv' not in adv_map:
        print(f"WARNING: Adversarial input not found: {adv_input_path}, skipping.")
        return

    adversarial = adv_map['adv']
    combined_importance, _, _, _ = compute_feature_importance(baseline, adversarial)

    print("\nFeature Importance Scores:")
    print("-" * 50)
    sorted_idx = np.argsort(combined_importance)[::-1]
    for i, idx in enumerate(sorted_idx, 1):
        print(f"{i:2d}. {FEATURE_NAMES[idx]:12s}: {combined_importance[idx]:.4f}")

    print("\nGenerating diagrams...")
    comparisons_dir = os.path.join(OUTPUT_DIR, 'comparisons')
    os.makedirs(comparisons_dir, exist_ok=True)

    plot_comprehensive_xai(
        baseline, adversarial, FEATURE_NAMES,
        model.upper(), attack.capitalize(),
        os.path.join(comparisons_dir, f'{output_prefix}_xai_analysis.png')
    )
    plot_feature_importance(
        combined_importance, FEATURE_NAMES,
        model.upper(), attack.capitalize(),
        os.path.join(comparisons_dir, f'{output_prefix}_feature_importance.png')
    )
    plot_perturbation_heatmap(
        baseline, adversarial, FEATURE_NAMES,
        model.upper(), attack.capitalize(),
        os.path.join(comparisons_dir, f'{output_prefix}_comparison.png')
    )
    plot_temporal_importance(
        baseline, adversarial, FEATURE_NAMES,
        model.upper(), attack.capitalize(),
        os.path.join(comparisons_dir, f'{output_prefix}_temporal_analysis.png')
    )


def main():
    parser = argparse.ArgumentParser(description='Generate XAI diagrams for Chronos and DeepAR')
    parser.add_argument('--model', type=str, default=None, help='Model name (e.g., chronos, deepar)')
    parser.add_argument('--attack', type=str, default=None, help='Attack type (e.g., latency, energy, bitflip)')
    parser.add_argument('--adv-input', type=str, default=None, help='Path to adversarial input .npy file')
    parser.add_argument('--baseline-csv', type=str, default='data/Location1.csv', help='Path to baseline CSV')
    parser.add_argument('--output-prefix', type=str, default=None, help='Output file prefix')

    args = parser.parse_args()

    if args.model and args.attack and args.adv_input:
        run_single(args.model, args.attack, args.adv_input, args.baseline_csv, args.output_prefix)
    else:
        inputs = discover_inputs()
        if not inputs:
            print("No adversarial inputs found in outputs/. Nothing to do.")
            return
        print(f"Auto-discovery: found {len(inputs)} adversarial inputs.\n")
        for model, attack, path in inputs:
            run_single(model, attack, path, args.baseline_csv)

    print("\n" + "=" * 70)
    print("XAI DIAGRAM GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

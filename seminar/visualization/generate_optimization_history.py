"""
Generate Optimization History Diagrams
======================================
Plots the evolution of attack metrics over generations/steps.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import argparse

from config import OUTPUT_DIR

def plot_history(npz_path, metric_key, ylabel, title, save_path, color='red'):
    if not os.path.exists(npz_path):
        print(f"Error: File {npz_path} not found.")
        return

    data = np.load(npz_path)

    # Check if keys exist
    if 'gen' not in data.files or metric_key not in data.files:
        print(f"Error: Missing keys in {npz_path}. Available: {data.files}")
        return

    generations = data['gen']
    metrics = data[metric_key]

    # Convert latency to ms if needed
    if 'latency' in metric_key.lower():
        metrics = metrics * 1000
        ylabel = ylabel.replace("(s)", "(ms)")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(generations, metrics, linewidth=2, color=color, label='Best Adversarial')

    ax.set_xlabel('Generation / Step', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Highlight start and end
    ax.scatter(generations[0], metrics[0], color='blue', zorder=5)
    ax.text(generations[0], metrics[0], f'{metrics[0]:.2f}', ha='right', va='bottom')

    ax.scatter(generations[-1], metrics[-1], color='blue', zorder=5)
    ax.text(generations[-1], metrics[-1], f'{metrics[-1]:.2f}', ha='left', va='top')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

METRIC_CONFIG = {
    'latency': ('best_latency', 'Latency (s)', '#d62728'),
    'energy':  ('best_power',   'Power (W)',    '#2ca02c'),
    'bitflip': ('best_flips',   'Bit-Flip Count', '#9467bd'),
}


def discover_npz_files():
    """Scan outputs/ for all *_generation_data.npz files and return (model, attack, path) tuples."""
    found = []
    for model_dir in ('deepar', 'act', 'chronos'):
        scan_dir = os.path.join(OUTPUT_DIR, model_dir)
        if not os.path.isdir(scan_dir):
            continue
        for fname in sorted(os.listdir(scan_dir)):
            if fname.endswith('_generation_data.npz'):
                prefix = fname.replace('_generation_data.npz', '')
                attack = prefix.replace(f'{model_dir}_', '', 1)
                found.append((model_dir, attack, os.path.join(scan_dir, fname)))
    return found


def run_single(model, attack):
    """Plot optimization history for a single model/attack."""
    comparisons_dir = os.path.join(OUTPUT_DIR, 'comparisons')
    os.makedirs(comparisons_dir, exist_ok=True)

    npz_path = os.path.join(OUTPUT_DIR, model, f'{model}_{attack}_generation_data.npz')
    if not os.path.exists(npz_path):
        print(f"File not found: {npz_path}, skipping.")
        return

    cfg = METRIC_CONFIG.get(attack)
    if cfg is None:
        print(f"No metric config for attack '{attack}', skipping.")
        return

    metric_key, ylabel, color = cfg
    title = f"{model.upper()} {attack.replace('_', ' ').title()} Attack Optimization"
    save_path = os.path.join(comparisons_dir, f"{model}_{attack}_optimization_history.png")

    plot_history(npz_path, metric_key, ylabel, title, save_path, color=color)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["chronos", "deepar", "act"], default=None)
    parser.add_argument("--attack", choices=["latency", "bitflip", "energy"], default=None)
    args = parser.parse_args()

    if args.model and args.attack:
        run_single(args.model, args.attack)
    else:
        discovered = discover_npz_files()
        # Filter to attacks that have metric config
        runnable = [(m, a, p) for m, a, p in discovered if a in METRIC_CONFIG]
        if not runnable:
            print("No generation data found in outputs/. Nothing to do.")
            return
        print(f"Auto-discovery: found {len(runnable)} optimization histories.\n")
        for model, attack, _ in runnable:
            run_single(model, attack)

if __name__ == "__main__":
    main()

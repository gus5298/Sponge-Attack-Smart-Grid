"""
Generate Optimization History Diagrams
======================================
Plots the evolution of attack metrics over generations/steps.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

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
    
    # Optional: Plot average fitness if available (scaled to match roughly?) 
    # Usually better to just show the target metric.
    
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["chronos", "deepar"], required=True)
    parser.add_argument("--attack", choices=["latency", "bitflip_blackbox", "bitflip_whitebox", "energy"], required=True)
    args = parser.parse_args()

    if args.model == "chronos":
        plot_history(
            "chronos_latency_generation_data.npz",
            "best_latency",
            "Latency (ms)",
            "Chronos Latency Attack Optimization",
            "chronos_latency_optimization_history.png",
            color="#d62728" # Red
        )
    elif args.model == "deepar":
        if args.attack == "energy":
            plot_history(
                "deepar_energy_generation_data.npz",
                "best_power",
                "Power (W)",
                "DeepAR Energy Attack Optimization",
                "deepar_energy_results.png", # Using the same name as attack script output for consistency
                color="#2ca02c" # Green
            )
        else:
            mode = args.attack.split("_")[1]
            plot_history(
                f"deepar_bitflip_{mode}_generation_data.npz",
                "best_flips",
                "Bit-Flip Count",
                f"DeepAR Bit-Flip ({mode.title()}) Optimization",
                f"deepar_{args.attack}_optimization_history.png",
                color="#ff7f0e" # Orange
            )

if __name__ == "__main__":
    main()

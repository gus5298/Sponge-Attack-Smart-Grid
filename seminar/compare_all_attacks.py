"""
Compare All Attacks (Latency or Energy)
=======================================
Generates a comparative analysis (Table + Plots) of the best adversarial inputs
found for Chronos, DeepAR, and ACT-LSTM models.

Usage:
    python compare_all_attacks.py --metric latency
    python compare_all_attacks.py --metric energy
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import CONTEXT_LEN, NUM_FEATURES, FEATURE_NAMES

METRIC_CONFIG = {
    'latency': {
        'models': {
            'Chronos': 'chronos_latency_best_input.npy',
            'DeepAR': 'deepar_latency_best_input.npy',
            'ACT-LSTM': 'act_latency_best_input.npy',
        },
        'results': {
            'Chronos': "Base: 230ms\nAdv: 230ms\nDelta: ~0%",
            'DeepAR': "Base: 50ms\nAdv: 53ms\nDelta: +6%",
            'ACT-LSTM': "Base: 225ms\nAdv: 384ms\nDelta: +70%",
        },
        'cmap': 'viridis',
        'table_title': 'Adversarial Input Statistics',
        'heatmap_suffix': 'Full Input Heatmap (Features x Time)',
        'output': 'attack_comparison_summary.png',
    },
    'energy': {
        'models': {
            'Chronos': 'chronos_energy_sponge_best_input.npy',
            'DeepAR': 'deepar_energy_best_input.npy',
            'ACT-LSTM': 'act_energy_best_input.npy',
        },
        'results': {
            'Chronos': "Base: 4.5J\nAdv: 4.5J\nDelta: ~0%",
            'DeepAR': "Base: 15.0J\nAdv: 16.2J\nDelta: +8%",
            'ACT-LSTM': "Base: 6.4J\nAdv: 8.9J\nDelta: +38%",
        },
        'cmap': 'inferno',
        'table_title': 'Energy-Adversarial Input Statistics',
        'heatmap_suffix': 'Energy Input Heatmap',
        'output': 'energy_attack_comparison_summary.png',
    },
}


def load_input(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Returning zeros.")
        return np.zeros((CONTEXT_LEN, NUM_FEATURES))
    return np.load(filepath).reshape(CONTEXT_LEN, NUM_FEATURES)


def plot_comparison(metric='latency'):
    cfg = METRIC_CONFIG[metric]
    models = cfg['models']
    results = cfg['results']

    plt.figure(figsize=(15, 12))

    # 1. Feature Statistics Table
    stats = []
    for model_name, path in models.items():
        data = load_input(path)
        row = {
            'Model': model_name,
            'Min Val': data.min(),
            'Max Val': data.max(),
            'Mean': data.mean(),
            'Std Dev': data.std(),
            'L2 Norm': np.linalg.norm(data),
        }
        stats.append(row)

    ax_table = plt.subplot(4, 1, 1)
    ax_table.axis('off')
    ax_table.set_title(cfg['table_title'])

    df = pd.DataFrame(stats)
    df_display = df.copy()
    for col in ['Mean', 'Std Dev', 'L2 Norm']:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.2e}")
    for col in ['Min Val', 'Max Val']:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.2e}")

    table = plt.table(cellText=df_display.values,
                      colLabels=df_display.columns,
                      cellLoc='center',
                      loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # 2. Heatmaps of full input features
    for i, (model_name, path) in enumerate(models.items()):
        data = load_input(path)
        plt.subplot(4, 1, i + 2)
        plt.imshow(data.T, cmap=cfg['cmap'], aspect='auto')
        plt.colorbar(label='Value')
        plt.title(f"{model_name} - {cfg['heatmap_suffix']}")
        plt.yticks(ticks=np.arange(NUM_FEATURES), labels=FEATURE_NAMES, rotation=0)
        plt.xlabel("Time Step")

        res_text = results.get(model_name, "")
        plt.text(1.25, 0.5, res_text, transform=plt.gca().transAxes,
                 verticalalignment='center', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.savefig(cfg['output'], dpi=150)
    print(f"Saved plot: {cfg['output']}")

    # Print Table
    print(f"\n{cfg['table_title']}:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare all attacks (latency or energy)')
    parser.add_argument('--metric', choices=['latency', 'energy'], default='latency',
                        help='Which metric to compare (default: latency)')
    args = parser.parse_args()
    plot_comparison(args.metric)

"""
Export Heatmaps to PDF (Latency or Energy)
==========================================
Generates separate PDF files for each model's heatmap with stats on the right.

Usage:
    python visualization/export_heatmaps_pdf.py --metric latency
    python visualization/export_heatmaps_pdf.py --metric energy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import matplotlib.pyplot as plt

from config import CONTEXT_LEN, NUM_FEATURES, FEATURE_NAMES, OUTPUT_DIR

METRIC_CONFIG = {
    'latency': {
        'models': {
            'Chronos': os.path.join(OUTPUT_DIR, 'chronos', 'chronos_latency_best_input.npy'),
            'DeepAR': os.path.join(OUTPUT_DIR, 'deepar', 'deepar_latency_best_input.npy'),
            'ACT-LSTM': os.path.join(OUTPUT_DIR, 'act', 'act_latency_best_input.npy'),
        },
        'results': {
            'Chronos': "Base: 230ms\nAdv: 230ms\nDelta: ~0%",
            'DeepAR': "Base: 50ms\nAdv: 53ms\nDelta: +6%",
            'ACT-LSTM': "Base: 225ms\nAdv: 384ms\nDelta: +70%",
        },
        'cmap': 'viridis',
        'title_suffix': 'Adversarial Input Heatmap (Features x Time)',
        'file_suffix': '_heatmap.pdf',
    },
    'energy': {
        'models': {
            'Chronos': os.path.join(OUTPUT_DIR, 'chronos', 'chronos_energy_best_input.npy'),
            'DeepAR': os.path.join(OUTPUT_DIR, 'deepar', 'deepar_energy_best_input.npy'),
            'ACT-LSTM': os.path.join(OUTPUT_DIR, 'act', 'act_energy_best_input.npy'),
        },
        'results': {
            'Chronos': "Base: 4.5J\nAdv: 4.5J\nDelta: ~0%",
            'DeepAR': "Base: 15.0J\nAdv: 16.2J\nDelta: +8%",
            'ACT-LSTM': "Base: 6.4J\nAdv: 8.9J\nDelta: +38%",
        },
        'cmap': 'inferno',
        'title_suffix': 'Energy Attack Heatmap (Features x Time)',
        'file_suffix': '_energy_heatmap.pdf',
    },
}


def load_input(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Returning zeros.")
        return np.zeros((CONTEXT_LEN, NUM_FEATURES))
    return np.load(filepath).reshape(CONTEXT_LEN, NUM_FEATURES)


def export_single_heatmap_pdf(model_name, path, output_filename, cmap, title_suffix, results):
    """Export a single heatmap with stats to a PDF file."""
    data = load_input(path)

    fig, ax = plt.subplots(figsize=(10, 4))

    im = ax.imshow(data.T, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax, label='Value')

    ax.set_title(f"{model_name} - {title_suffix}", fontsize=14, fontweight='bold')
    ax.set_yticks(np.arange(NUM_FEATURES))
    ax.set_yticklabels(FEATURE_NAMES, rotation=0)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)

    res_text = results.get(model_name, "")
    ax.text(1.25, 0.5, res_text, transform=ax.transAxes,
            verticalalignment='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.9))

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)

    plt.savefig(output_filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_filename}")


def main():
    parser = argparse.ArgumentParser(description='Export heatmaps to PDF (latency or energy)')
    parser.add_argument('--metric', choices=['latency', 'energy'], default='latency',
                        help='Which metric to export (default: latency)')
    args = parser.parse_args()

    cfg = METRIC_CONFIG[args.metric]
    print(f"Exporting {args.metric} heatmaps to PDF...\n")

    comparisons_dir = os.path.join(OUTPUT_DIR, 'comparisons')
    os.makedirs(comparisons_dir, exist_ok=True)

    for model_name, path in cfg['models'].items():
        safe_name = model_name.lower().replace('-', '_').replace(' ', '_')
        output_filename = os.path.join(comparisons_dir, f"{safe_name}{cfg['file_suffix']}")
        export_single_heatmap_pdf(
            model_name, path, output_filename,
            cfg['cmap'], cfg['title_suffix'], cfg['results']
        )

    print(f"\nAll {args.metric} heatmaps exported successfully!")
    print("\nGenerated files:")
    for model_name in cfg['models']:
        safe_name = model_name.lower().replace('-', '_').replace(' ', '_')
        print(f"  - {os.path.join(comparisons_dir, safe_name + cfg['file_suffix'])}")


if __name__ == "__main__":
    main()

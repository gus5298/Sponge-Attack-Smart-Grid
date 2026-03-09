import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import matplotlib.pyplot as plt

from config import CONTEXT_LEN, NUM_FEATURES, MAX_PONDERS, ALL_FEATURES, OUTPUT_DIR
from utils.model_loader import load_act, load_seed_from_csv, get_device
from utils.xai_helpers import load_adversarial_inputs

print("=" * 70)
print("ACT PONDER STEP VISUALIZATION")
print("=" * 70)

device = get_device()

comparisons_dir = os.path.join(OUTPUT_DIR, 'comparisons')
os.makedirs(comparisons_dir, exist_ok=True)


class ACTModelWithPonderTracking(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.ponder_history = []

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, self.base_model.hidden_size, device=x.device)
        c = torch.zeros(batch_size, self.base_model.hidden_size, device=x.device)
        ponder_per_step = []
        for t in range(seq_len):
            input_t = x[:, t, :]
            (h, c), ponders = self.base_model.act_cell(input_t, (h, c))
            ponder_per_step.append(ponders.item())
        self.ponder_history = ponder_per_step
        out = self.base_model.fc(h)
        return out, sum(ponder_per_step) / seq_len


def get_ponder_distribution(tracked_model, input_data):
    x = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        _, avg_ponder = tracked_model(x)
    return tracked_model.ponder_history, avg_ponder


if __name__ == "__main__":
    baseline_data, _, _ = load_seed_from_csv()

    base_model = load_act(device=device)
    tracked_model = ACTModelWithPonderTracking(base_model)

    adv_inputs = load_adversarial_inputs({
        'ACT PGD Energy': os.path.join(OUTPUT_DIR, 'act', 'act_pgd_energy_best_input.npy'),
        'ACT PGD Latency': os.path.join(OUTPUT_DIR, 'act', 'act_pgd_latency_best_input.npy'),
        'ACT GA Energy': os.path.join(OUTPUT_DIR, 'act', 'act_energy_best_input.npy'),
        'ACT GA Latency': os.path.join(OUTPUT_DIR, 'act', 'act_latency_best_input.npy'),
    })

    print("\nAnalyzing ponder steps...")

    baseline_ponders, baseline_avg = get_ponder_distribution(tracked_model, baseline_data)
    print(f"\nBaseline: Avg={baseline_avg:.2f} ponders/step, Total={sum(baseline_ponders):.0f}")

    all_ponders = {'Baseline': baseline_ponders}
    all_avgs = {'Baseline': baseline_avg}

    for name, adv_data in adv_inputs.items():
        ponders, avg = get_ponder_distribution(tracked_model, adv_data)
        all_ponders[name] = ponders
        all_avgs[name] = avg
        change = (avg / baseline_avg - 1) * 100
        print(f"{name}: Avg={avg:.2f} ponders/step ({change:+.1f}%), Total={sum(ponders):.0f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for name, ponders in all_ponders.items():
        alpha = 1.0 if name == 'Baseline' else 0.7
        lw = 2 if name == 'Baseline' else 1.5
        axes[0, 0].plot(ponders, label=name, alpha=alpha, linewidth=lw)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Ponder Steps')
    axes[0, 0].set_title('Ponder Steps per Time Step')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=MAX_PONDERS, color='red', linestyle='--', alpha=0.5)

    names = list(all_avgs.keys())
    avgs = [all_avgs[n] for n in names]
    colors = ['steelblue' if n == 'Baseline' else 'coral' for n in names]
    bars = axes[0, 1].bar(names, avgs, color=colors)
    axes[0, 1].set_ylabel('Average Ponder Steps')
    axes[0, 1].set_title('Average Ponder Steps Comparison')
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for bar, avg in zip(bars, avgs):
        axes[0, 1].annotate(f'{avg:.2f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                            ha='center', va='bottom', fontsize=9)

    if 'ACT PGD Energy' in all_ponders:
        diff = np.array(all_ponders['ACT PGD Energy']) - np.array(baseline_ponders)
        bar_colors = ['coral' if d > 0 else 'steelblue' for d in diff]
        axes[1, 0].bar(range(len(diff)), diff, color=bar_colors, alpha=0.7)
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Ponder Difference')
        axes[1, 0].set_title('Ponder Increase (Adversarial - Baseline)')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[1, 0].grid(True, alpha=0.3, axis='y')

    for name, ponders in all_ponders.items():
        axes[1, 1].hist(ponders, bins=range(1, MAX_PONDERS+2), alpha=0.5, label=name, density=True)
    axes[1, 1].set_xlabel('Ponder Steps')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Ponder Step Distribution')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(comparisons_dir, 'act_ponder_visualization.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()

    if adv_inputs:
        first_name = list(adv_inputs.keys())[0]
        first_adv = adv_inputs[first_name]
        perturbation = first_adv - baseline_data
        ponder_change = np.array(all_ponders[first_name]) - np.array(baseline_ponders)

        fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

        step_perturb = np.abs(perturbation).sum(axis=1)
        correlation = np.corrcoef(step_perturb, ponder_change)[0, 1]
        axes2[0].scatter(step_perturb, ponder_change, alpha=0.6, c=range(CONTEXT_LEN), cmap='viridis')
        axes2[0].set_xlabel('Perturbation Magnitude (per step)')
        axes2[0].set_ylabel('Ponder Change')
        axes2[0].set_title(f'Perturbation vs Ponder Change (r={correlation:.3f})')
        axes2[0].grid(True, alpha=0.3)

        feature_ponder_corr = []
        for f in range(NUM_FEATURES):
            corr = np.corrcoef(np.abs(perturbation[:, f]), ponder_change)[0, 1]
            feature_ponder_corr.append(corr if not np.isnan(corr) else 0)

        y = np.arange(NUM_FEATURES)
        bar_colors = ['coral' if c > 0 else 'steelblue' for c in feature_ponder_corr]
        axes2[1].barh(y, feature_ponder_corr, color=bar_colors, alpha=0.7)
        axes2[1].set_yticks(y)
        axes2[1].set_yticklabels(ALL_FEATURES, fontsize=8)
        axes2[1].set_xlabel('Correlation with Ponder Change')
        axes2[1].set_title('Feature-Ponder Correlation')
        axes2[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        axes2[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        save_path2 = os.path.join(comparisons_dir, 'act_ponder_correlation.png')
        plt.savefig(save_path2, dpi=150)
        print(f"Saved: {save_path2}")
        plt.close()

    print("\n" + "=" * 70)
    print("ACT PONDER ANALYSIS COMPLETE")
    print("=" * 70)

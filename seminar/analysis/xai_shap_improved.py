import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import shap

from config import CONTEXT_LEN, NUM_FEATURES, ALL_FEATURES, OUTPUT_DIR
from utils.model_loader import load_act, load_deepar, load_seed_from_csv, get_device
from utils.xai_helpers import (load_adversarial_inputs, act_energy_proxy, deepar_energy_proxy)

print("=" * 70)
print("IMPROVED SHAP ANALYSIS")
print("=" * 70)

device = get_device()


def run_shap_per_feature(model, model_name, predict_fn, baseline_data, adv_data, feature_names):
    print(f"\nRunning SHAP for {model_name}...")

    feature_means = baseline_data.mean(axis=0)
    feature_means_adv = adv_data.mean(axis=0)

    def predict_feature_means(x):
        results = []
        for sample in x:
            full_input = np.tile(sample, (CONTEXT_LEN, 1)).astype(np.float32)
            results.append(predict_fn(full_input))
        return np.array(results)

    background = feature_means.reshape(1, -1) + np.random.normal(0, 0.1, (30, NUM_FEATURES))
    explainer = shap.KernelExplainer(predict_feature_means, background)

    shap_baseline = explainer.shap_values(feature_means.reshape(1, -1), nsamples=100)
    shap_adv = explainer.shap_values(feature_means_adv.reshape(1, -1), nsamples=100)

    return shap_baseline.flatten(), shap_adv.flatten()


def create_shap_summary(act_model, deepar_model, baseline_data, adv_data, feature_names):
    comparisons_dir = os.path.join(OUTPUT_DIR, 'comparisons')
    os.makedirs(comparisons_dir, exist_ok=True)

    def act_predict(x):
        return act_energy_proxy(x, act_model, device)[0]

    def deepar_predict(x):
        return deepar_energy_proxy(x, deepar_model, device)[0]

    print("\nComputing SHAP for ACT baseline vs adversarial...")
    act_shap_base, act_shap_adv = run_shap_per_feature(
        act_model, "ACT", act_predict, baseline_data, adv_data['ACT PGD Energy'], feature_names
    )

    print("\nComputing SHAP for DeepAR baseline vs adversarial...")
    deepar_shap_base, deepar_shap_adv = run_shap_per_feature(
        deepar_model, "DeepAR", deepar_predict, baseline_data, adv_data['DeepAR PGD Energy'], feature_names
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    y = np.arange(len(feature_names))
    width = 0.35

    for ax, shap_base, shap_adv, title in [
        (axes[0, 0], act_shap_base, act_shap_adv, 'ACT-LSTM'),
        (axes[0, 1], deepar_shap_base, deepar_shap_adv, 'DeepAR-LSTM'),
    ]:
        ax.barh(y - width/2, shap_base, width, label='Baseline', color='steelblue', alpha=0.8)
        ax.barh(y + width/2, shap_adv, width, label='Adversarial', color='coral', alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(feature_names, fontsize=8)
        ax.set_xlabel('SHAP Value')
        ax.set_title(f'{title}: Feature SHAP (Baseline vs Adversarial)')
        ax.legend()
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

    for ax, shap_base, shap_adv, title in [
        (axes[1, 0], act_shap_base, act_shap_adv, 'ACT-LSTM'),
        (axes[1, 1], deepar_shap_base, deepar_shap_adv, 'DeepAR-LSTM'),
    ]:
        diff = shap_adv - shap_base
        colors = ['coral' if d > 0 else 'steelblue' for d in diff]
        ax.barh(y, diff, color=colors, alpha=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(feature_names, fontsize=8)
        ax.set_xlabel('\u0394SHAP (Adversarial - Baseline)')
        ax.set_title(f'{title}: SHAP Change Due to Attack')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    save_path = os.path.join(comparisons_dir, 'shap_improved_analysis.png')
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved: {save_path}")
    plt.close()

    fig2, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(feature_names))
    width = 0.35
    ax.bar(x - width/2, np.abs(act_shap_adv), width, label='ACT-LSTM', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, np.abs(deepar_shap_adv), width, label='DeepAR-LSTM', color='coral', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('|SHAP Value|')
    ax.set_title('Feature Importance Comparison: ACT vs DeepAR (on Adversarial Inputs)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_path2 = os.path.join(comparisons_dir, 'shap_model_comparison.png')
    plt.savefig(save_path2, dpi=150)
    print(f"Saved: {save_path2}")
    plt.close()


if __name__ == "__main__":
    baseline_data, _, _ = load_seed_from_csv()

    act_model = load_act(device=device)
    deepar_model, _ = load_deepar(device=device)

    adv_data = load_adversarial_inputs({
        'ACT PGD Energy': os.path.join(OUTPUT_DIR, 'act', 'act_pgd_energy_best_input.npy'),
        'DeepAR PGD Energy': os.path.join(OUTPUT_DIR, 'deepar', 'deepar_pgd_energy_best_input.npy'),
    })

    if len(adv_data) >= 2:
        create_shap_summary(act_model, deepar_model, baseline_data, adv_data, ALL_FEATURES)

    print("\n" + "=" * 70)
    print("IMPROVED SHAP ANALYSIS COMPLETE")
    print("=" * 70)

import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, FeatureAblation, Saliency

from config import CONTEXT_LEN, NUM_FEATURES, ALL_FEATURES
from utils.model_loader import load_act, load_deepar, load_seed_from_csv, get_device
from utils.xai_helpers import ACTEnergyWrapper, ACTLatencyWrapper, DeepAREnergyWrapper
from utils.visualization import plot_attribution_heatmap

print("=" * 70)
print("XAI ANALYSIS OF PGD ADVERSARIAL ATTACKS")
print("=" * 70)

device = get_device()


def compute_integrated_gradients(wrapper, input_tensor, baseline_tensor, n_steps=50):
    ig = IntegratedGradients(wrapper)
    attributions = ig.attribute(input_tensor, baselines=baseline_tensor, n_steps=n_steps)
    return attributions.squeeze(0).cpu().numpy()


def plot_xai_analysis(attributions, perturbations, feature_names, title_prefix, save_path):
    from matplotlib.colors import TwoSlopeNorm
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    plot_attribution_heatmap(attributions, feature_names, f'{title_prefix}: Integrated Gradients Attribution', axes[0, 0])

    vmax_p = np.abs(perturbations).max()
    if vmax_p == 0:
        vmax_p = 1
    norm_p = TwoSlopeNorm(vmin=-vmax_p, vcenter=0, vmax=vmax_p)
    im2 = axes[0, 1].imshow(perturbations.T, aspect='auto', cmap='RdBu_r', norm=norm_p)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Feature')
    axes[0, 1].set_yticks(range(len(feature_names)))
    axes[0, 1].set_yticklabels(feature_names, fontsize=8)
    axes[0, 1].set_title(f'{title_prefix}: PGD Perturbation Heatmap')
    plt.colorbar(im2, ax=axes[0, 1])

    temporal_importance = np.abs(attributions).sum(axis=1)
    temporal_perturbation = np.abs(perturbations).sum(axis=1)
    x = np.arange(len(temporal_importance))
    width = 0.35
    axes[1, 0].bar(x - width/2, temporal_importance, width, label='IG Attribution', color='steelblue', alpha=0.8)
    axes[1, 0].bar(x + width/2, temporal_perturbation, width, label='Perturbation Mag', color='coral', alpha=0.8)
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Importance / Magnitude')
    axes[1, 0].set_title(f'{title_prefix}: Temporal Importance')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    feature_importance = np.abs(attributions).sum(axis=0)
    feature_perturbation = np.abs(perturbations).sum(axis=0)
    y = np.arange(len(feature_names))
    axes[1, 1].barh(y - 0.2, feature_importance, 0.4, label='IG Attribution', color='steelblue', alpha=0.8)
    axes[1, 1].barh(y + 0.2, feature_perturbation, 0.4, label='Perturbation Mag', color='coral', alpha=0.8)
    axes[1, 1].set_xlabel('Importance / Magnitude')
    axes[1, 1].set_ylabel('Feature')
    axes[1, 1].set_yticks(y)
    axes[1, 1].set_yticklabels(feature_names, fontsize=8)
    axes[1, 1].set_title(f'{title_prefix}: Feature Importance')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def plot_comparison(baseline_attr, adv_attr, feature_names, title_prefix, save_path):
    from matplotlib.colors import TwoSlopeNorm
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmax = max(np.abs(baseline_attr).max(), np.abs(adv_attr).max())
    if vmax == 0:
        vmax = 1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for ax, data, subtitle in [(axes[0], baseline_attr, 'Baseline'), (axes[1], adv_attr, 'Adversarial')]:
        im = ax.imshow(data.T, aspect='auto', cmap='RdBu_r', norm=norm)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Feature')
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=8)
        ax.set_title(f'{title_prefix}: {subtitle} Attribution')
        plt.colorbar(im, ax=ax)

    diff = adv_attr - baseline_attr
    vmax_d = np.abs(diff).max()
    if vmax_d == 0:
        vmax_d = 1
    norm_d = TwoSlopeNorm(vmin=-vmax_d, vcenter=0, vmax=vmax_d)
    im3 = axes[2].imshow(diff.T, aspect='auto', cmap='RdBu_r', norm=norm_d)
    axes[2].set_xlabel('Time Step')
    axes[2].set_ylabel('Feature')
    axes[2].set_yticks(range(len(feature_names)))
    axes[2].set_yticklabels(feature_names, fontsize=8)
    axes[2].set_title(f'{title_prefix}: Attribution Difference')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()


def analyze_attack(model_name, wrapper, baseline_data, adv_path, feature_names):
    print(f"\n{'='*50}")
    print(f"Analyzing {model_name}")
    print(f"{'='*50}")

    adv_data = np.load(adv_path).reshape(CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
    baseline_tensor = torch.tensor(baseline_data, dtype=torch.float32).unsqueeze(0).to(device)
    adv_tensor = torch.tensor(adv_data, dtype=torch.float32).unsqueeze(0).to(device)
    zero_baseline = torch.zeros_like(baseline_tensor)

    print("Computing Integrated Gradients for adversarial input...")
    adv_attr = compute_integrated_gradients(wrapper, adv_tensor, zero_baseline)
    print("Computing Integrated Gradients for baseline input...")
    baseline_attr = compute_integrated_gradients(wrapper, baseline_tensor, zero_baseline)

    perturbation = adv_data - baseline_data

    print("Generating visualizations...")
    plot_xai_analysis(adv_attr, perturbation, feature_names,
                      model_name, f"{model_name.lower().replace(' ', '_')}_xai_analysis.png")
    plot_comparison(baseline_attr, adv_attr, feature_names,
                   model_name, f"{model_name.lower().replace(' ', '_')}_attribution_comparison.png")

    feature_importance = np.abs(adv_attr).sum(axis=0)
    feature_perturbation = np.abs(perturbation).sum(axis=0)

    print("\nTop 5 Most Important Features (by IG attribution):")
    for i, idx in enumerate(np.argsort(feature_importance)[::-1][:5]):
        print(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")

    print("\nTop 5 Most Perturbed Features:")
    for i, idx in enumerate(np.argsort(feature_perturbation)[::-1][:5]):
        print(f"  {i+1}. {feature_names[idx]}: {feature_perturbation[idx]:.4f}")

    correlation = np.corrcoef(np.abs(adv_attr).flatten(), np.abs(perturbation).flatten())[0, 1]
    print(f"\nCorrelation between attribution and perturbation: {correlation:.4f}")

    return {'feature_importance': feature_importance, 'feature_perturbation': feature_perturbation}


if __name__ == "__main__":
    baseline_data, _, _ = load_seed_from_csv()
    results = {}

    attack_configs = [
        ("ACT PGD Energy", ACTEnergyWrapper, load_act, "act_pgd_energy_best_input.npy"),
        ("ACT PGD Latency", ACTLatencyWrapper, load_act, "act_pgd_latency_best_input.npy"),
        ("DeepAR PGD Energy", DeepAREnergyWrapper, lambda **kw: load_deepar(**kw)[0], "deepar_pgd_energy_best_input.npy"),
        ("DeepAR PGD Latency", DeepAREnergyWrapper, lambda **kw: load_deepar(**kw)[0], "deepar_pgd_latency_best_input.npy"),
    ]

    for name, wrapper_cls, loader, adv_path in attack_configs:
        try:
            model = loader(device=device)
            wrapper = wrapper_cls(model)
            results[name] = analyze_attack(name, wrapper, baseline_data, adv_path, ALL_FEATURES)
        except FileNotFoundError:
            print(f"{name} results not found, skipping...")

    if len(results) > 1:
        print("\n" + "=" * 70)
        print("CROSS-MODEL COMPARISON")
        print("=" * 70)

        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
        if len(results) == 1:
            axes = [axes]

        for ax, (name, res) in zip(axes, results.items()):
            y = np.arange(len(ALL_FEATURES))
            ax.barh(y, res['feature_importance'], color='steelblue', alpha=0.8)
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.set_yticks(y)
            ax.set_yticklabels(ALL_FEATURES, fontsize=8)
            ax.set_title(f'{name}\nFeature Importance')
            ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig('xai_cross_model_comparison.png', dpi=150)
        print("Saved: xai_cross_model_comparison.png")
        plt.close()

    print("\n" + "=" * 70)
    print("XAI ANALYSIS COMPLETE")
    print("=" * 70)

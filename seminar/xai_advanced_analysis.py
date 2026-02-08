import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import shap

from config import CONTEXT_LEN, NUM_FEATURES, ALL_FEATURES
from utils.model_loader import load_act, load_deepar, load_seed_from_csv, make_predictor, get_device
from utils.power_monitor import PowerMonitor
from utils.metrics import measure_energy
from utils.xai_helpers import (load_adversarial_inputs, act_energy_proxy, deepar_energy_proxy)
from utils.visualization import plot_attribution_heatmap, plot_feature_importance_barh

print("=" * 70)
print("ADVANCED XAI ANALYSIS: SHAP, TRANSFERABILITY, FREQUENCY")
print("=" * 70)

device = get_device()
power_monitor = PowerMonitor(0.01)


def run_shap_analysis(model, model_name, proxy_fn, baseline_data, adv_data, feature_names):
    print(f"\n{'='*50}")
    print(f"SHAP Analysis: {model_name}")
    print(f"{'='*50}")

    flat_baseline = baseline_data.flatten().reshape(1, -1)
    flat_adv = adv_data.flatten().reshape(1, -1)
    background = flat_baseline + np.random.normal(0, 0.1, (50, flat_baseline.shape[1]))

    def predict_fn(x):
        results = []
        for sample in x:
            reshaped = sample.reshape(1, CONTEXT_LEN, NUM_FEATURES).astype(np.float32)
            results.append(proxy_fn(reshaped, model, device)[0])
        return np.array(results)

    print("Computing SHAP values (this may take a minute)...")
    explainer = shap.KernelExplainer(predict_fn, background[:20])
    shap_values = explainer.shap_values(flat_adv, nsamples=100)
    shap_2d = shap_values.reshape(CONTEXT_LEN, NUM_FEATURES)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_attribution_heatmap(shap_2d, feature_names, f'{model_name}: SHAP Values Heatmap', axes[0])
    feature_importance = np.abs(shap_2d).sum(axis=0)
    plot_feature_importance_barh(feature_importance, feature_names, f'{model_name}: Feature Importance (SHAP)', axes[1])
    plt.tight_layout()
    save_path = f"{model_name.lower().replace(' ', '_').replace('-', '_')}_shap_analysis.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

    print("\nTop 5 Most Important Features (SHAP):")
    sorted_idx = np.argsort(feature_importance)[::-1]
    for i, idx in enumerate(sorted_idx[:5]):
        print(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")

    return shap_2d, feature_importance


def run_transferability_analysis(act_model, deepar_model, baseline_data,
                                  act_adv_energy, act_adv_latency,
                                  deepar_adv_energy, deepar_adv_latency):
    print("\n" + "=" * 70)
    print("ATTACK TRANSFERABILITY ANALYSIS")
    print("=" * 70)

    act_predict = make_predictor(act_model, device)
    deepar_predict = make_predictor(deepar_model, device)

    results = {'scenario': [], 'act_energy': [], 'act_latency': [],
               'deepar_energy': [], 'deepar_latency': []}

    inputs = {
        'Baseline': baseline_data,
        'ACT Energy Adv': act_adv_energy,
        'ACT Latency Adv': act_adv_latency,
        'DeepAR Energy Adv': deepar_adv_energy,
        'DeepAR Latency Adv': deepar_adv_latency,
    }

    for name, inp in inputs.items():
        if inp is None:
            continue
        act_stats = measure_energy(act_predict, inp, power_monitor, device, 20)
        deepar_stats = measure_energy(deepar_predict, inp, power_monitor, device, 20)
        results['scenario'].append(name)
        results['act_energy'].append(act_stats['energy_per_inference'] * 1000)
        results['act_latency'].append(act_stats['latency'] * 1000)
        results['deepar_energy'].append(deepar_stats['energy_per_inference'] * 1000)
        results['deepar_latency'].append(deepar_stats['latency'] * 1000)
        print(f"\n{name}:")
        print(f"  ACT:    Energy={act_stats['energy_per_inference']*1000:.2f}mJ, Latency={act_stats['latency']*1000:.2f}ms")
        print(f"  DeepAR: Energy={deepar_stats['energy_per_inference']*1000:.2f}mJ, Latency={deepar_stats['latency']*1000:.2f}ms")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x = np.arange(len(results['scenario']))
    width = 0.35

    for ax, act_key, deepar_key, ylabel, title in [
        (axes[0, 0], 'act_energy', 'deepar_energy', 'Energy (mJ)', 'Energy: Cross-Model Attack Transferability'),
        (axes[0, 1], 'act_latency', 'deepar_latency', 'Latency (ms)', 'Latency: Cross-Model Attack Transferability'),
    ]:
        ax.bar(x - width/2, results[act_key], width, label='ACT', color='steelblue')
        ax.bar(x + width/2, results[deepar_key], width, label='DeepAR', color='coral')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(results['scenario'], rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    for ax, act_key, deepar_key, ylabel, title in [
        (axes[1, 0], 'act_energy', 'deepar_energy', 'Energy Change (%)', 'Energy Change from Baseline'),
        (axes[1, 1], 'act_latency', 'deepar_latency', 'Latency Change (%)', 'Latency Change from Baseline'),
    ]:
        act_change = [(v/results[act_key][0] - 1)*100 for v in results[act_key]]
        deepar_change = [(v/results[deepar_key][0] - 1)*100 for v in results[deepar_key]]
        ax.bar(x - width/2, act_change, width, label='ACT', color='steelblue')
        ax.bar(x + width/2, deepar_change, width, label='DeepAR', color='coral')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(results['scenario'], rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('attack_transferability_analysis.png', dpi=150)
    print("\nSaved: attack_transferability_analysis.png")
    plt.close()
    return results


def run_frequency_analysis(baseline_data, adv_inputs, feature_names):
    print("\n" + "=" * 70)
    print("FREQUENCY DOMAIN ANALYSIS")
    print("=" * 70)

    perturbations = {}
    for name, adv_data in adv_inputs.items():
        if adv_data is not None:
            perturbation = adv_data - baseline_data
            perturbations[name] = perturbation
            fft_magnitude = np.abs(np.fft.fft2(perturbation))
            print(f"\n{name}:")
            print(f"  Max perturbation: {np.abs(perturbation).max():.4f}")
            print(f"  Mean perturbation: {np.abs(perturbation).mean():.4f}")
            print(f"  Dominant frequency component: {np.unravel_index(fft_magnitude[1:, 1:].argmax(), fft_magnitude[1:, 1:].shape)}")

    if not perturbations:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    first_name = list(perturbations.keys())[0]
    first_perturb = perturbations[first_name]

    temporal_perturb = first_perturb.mean(axis=1)
    freqs = np.fft.fftfreq(len(temporal_perturb))
    fft_temporal = np.abs(np.fft.fft(temporal_perturb))

    axes[0, 0].plot(temporal_perturb, 'b-', alpha=0.7)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Mean Perturbation')
    axes[0, 0].set_title(f'{first_name}: Temporal Perturbation Pattern')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].stem(freqs[:len(freqs)//2], fft_temporal[:len(fft_temporal)//2], basefmt=' ')
    axes[0, 1].set_xlabel('Frequency')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].set_title(f'{first_name}: Frequency Spectrum')
    axes[0, 1].grid(True, alpha=0.3)

    feature_power = {}
    for name, perturb in perturbations.items():
        feature_power[name] = [np.abs(np.fft.fft(perturb[:, f]))**2 for f in range(NUM_FEATURES)]
        feature_power[name] = [p.sum() for p in feature_power[name]]

    x_feat = np.arange(NUM_FEATURES)
    bar_width = 0.8 / len(perturbations)
    colors = plt.cm.tab10(np.linspace(0, 1, len(perturbations)))
    for i, (name, power) in enumerate(feature_power.items()):
        offset = (i - len(perturbations)/2 + 0.5) * bar_width
        axes[1, 0].bar(x_feat + offset, power, bar_width, label=name, color=colors[i])
    axes[1, 0].set_xlabel('Feature')
    axes[1, 0].set_ylabel('Spectral Power')
    axes[1, 0].set_title('Per-Feature Spectral Power')
    axes[1, 0].set_xticks(x_feat)
    axes[1, 0].set_xticklabels(feature_names, rotation=45, ha='right', fontsize=7)
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    for name, perturb in perturbations.items():
        axes[1, 1].hist(perturb.flatten(), bins=50, alpha=0.5, label=name, density=True)
    axes[1, 1].set_xlabel('Perturbation Value')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Perturbation Distribution')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('frequency_analysis.png', dpi=150)
    print("\nSaved: frequency_analysis.png")
    plt.close()

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    for i, (name, perturb) in enumerate(list(perturbations.items())[:4]):
        ax = axes2[i//2, i%2]
        fft_shifted = np.fft.fftshift(np.abs(np.fft.fft2(perturb)))
        im = ax.imshow(np.log1p(fft_shifted.T), aspect='auto', cmap='viridis')
        ax.set_xlabel('Time Frequency')
        ax.set_ylabel('Feature Frequency')
        ax.set_title(f'{name}: 2D FFT Magnitude (log scale)')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('frequency_2d_fft.png', dpi=150)
    print("Saved: frequency_2d_fft.png")
    plt.close()


if __name__ == "__main__":
    baseline_data, _, _ = load_seed_from_csv()

    act_model = load_act(device=device)
    deepar_model, _ = load_deepar(device=device)

    adv_inputs = load_adversarial_inputs({
        'ACT PGD Energy': "act_pgd_energy_best_input.npy",
        'ACT PGD Latency': "act_pgd_latency_best_input.npy",
        'DeepAR PGD Energy': "deepar_pgd_energy_best_input.npy",
        'DeepAR PGD Latency': "deepar_pgd_latency_best_input.npy",
    })

    act_adv_energy = adv_inputs.get('ACT PGD Energy')
    act_adv_latency = adv_inputs.get('ACT PGD Latency')
    deepar_adv_energy = adv_inputs.get('DeepAR PGD Energy')
    deepar_adv_latency = adv_inputs.get('DeepAR PGD Latency')

    print("\n" + "=" * 70)
    print("PART 1: SHAP ANALYSIS")
    print("=" * 70)

    if act_adv_energy is not None:
        run_shap_analysis(act_model, "ACT-LSTM", act_energy_proxy, baseline_data, act_adv_energy, ALL_FEATURES)
    if deepar_adv_energy is not None:
        run_shap_analysis(deepar_model, "DeepAR-LSTM", deepar_energy_proxy, baseline_data, deepar_adv_energy, ALL_FEATURES)

    print("\n" + "=" * 70)
    print("PART 2: ATTACK TRANSFERABILITY")
    print("=" * 70)

    run_transferability_analysis(act_model, deepar_model, baseline_data,
                                  act_adv_energy, act_adv_latency,
                                  deepar_adv_energy, deepar_adv_latency)

    print("\n" + "=" * 70)
    print("PART 3: FREQUENCY ANALYSIS")
    print("=" * 70)

    run_frequency_analysis(baseline_data, adv_inputs, ALL_FEATURES)

    print("\n" + "=" * 70)
    print("ADVANCED XAI ANALYSIS COMPLETE")
    print("=" * 70)

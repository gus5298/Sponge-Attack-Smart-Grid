"""
Generate Metric Comparison Diagrams
===================================
Generates 'Baseline vs Adversarial' bar charts for Latency.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import os

from config import CONTEXT_LEN, PREDICTION_LEN, NUM_FEATURES, DATA_PATH
from utils.model_loader import load_deepar, load_seed, make_predictor, get_device
from utils.data_loader import load_seed_data
from utils.metrics import measure_latency, measure_energy
from utils.power_monitor import PowerMonitor

def get_chronos_metrics():
    """Calculates Chronos latency metrics."""
    print("Loading Chronos for metric calculation...")
    from chronos import ChronosPipeline
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-base",
        device_map=device,
        torch_dtype=torch.float32,
    )
    
    seed_data, _, _ = load_seed_data(DATA_PATH, CONTEXT_LEN)
    
    if os.path.exists("chronos_latency_best_input.npy"):
        adv_data = np.load("chronos_latency_best_input.npy").reshape(CONTEXT_LEN, NUM_FEATURES)
    else:
        print("Warning: Adv input not found, using random")
        adv_data = np.random.randn(CONTEXT_LEN, NUM_FEATURES)

    def make_prediction(input_array):
        if len(input_array.shape) == 2:
            univariate = input_array[:, 0]
        else:
            univariate = input_array
        inp = torch.tensor(univariate, dtype=torch.float32)
        pipeline.predict(inp, prediction_length=PREDICTION_LEN)
        
    monitor = PowerMonitor()
    
    print("Measuring Baseline Latency...")
    base_stats = measure_latency(make_prediction, seed_data, monitor, device, num_reps=20)
    
    print("Measuring Adversarial Latency...")
    adv_stats = measure_latency(make_prediction, adv_data, monitor, device, num_reps=20)
    
    return {
        'Baseline': base_stats['latency'] * 1000, 
        'Adversarial': adv_stats['latency'] * 1000
    }

def get_deepar_metrics(mode="blackbox"):
    """Calculates DeepAR Energy metrics."""
    print(f"Calculating DeepAR {mode} Energy...")

    device = get_device()
    model, checkpoint = load_deepar(device=device)
    seed_data, _, _ = load_seed(checkpoint=checkpoint)

    if mode == "energy":
        filename = "deepar_energy_best_input.npy"
    else:
        filename = f"deepar_bitflip_{mode}_best_input.npy"

    if os.path.exists(filename):
        adv_data = np.load(filename).reshape(CONTEXT_LEN, NUM_FEATURES)
    else:
        print(f"Warning: {filename} not found")
        adv_data = seed_data

    make_prediction = make_predictor(model, device)

    # Use real PowerMonitor (CPU/GPU)
    monitor = PowerMonitor(sample_interval=0.01)
    
    print("Measuring Baseline Energy...")
    # Using multiple reps to capture enough samples for power monitoring
    base_stats = measure_energy(make_prediction, seed_data, monitor, device, num_reps=50)
    base_readings = list(monitor.readings)
    
    print("Measuring Adversarial Energy...")
    adv_stats = measure_energy(make_prediction, adv_data, monitor, device, num_reps=50)
    adv_readings = list(monitor.readings)
    
    # Plot Power Trace if requested
    plot_power_trace(base_readings, adv_readings, mode, f"deepar_bitflip_{mode}_power_trace.png")
    
    # Return Energy in mJ (Joules * 1000)
    return {
        'Baseline': base_stats['energy_per_inference'] * 1000,
        'Adversarial': adv_stats['energy_per_inference'] * 1000
    }

def plot_power_trace(base_readings, adv_readings, attack_mode, save_path):
    """Plots power consumption over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    b_time = np.array([r['timestamp'] for r in base_readings])
    b_time -= b_time[0] # Relative time
    b_power = np.array([r['gpu_power'] if r['gpu_power'] > 0 else r['cpu_power'] for r in base_readings])
    
    a_time = np.array([r['timestamp'] for r in adv_readings])
    a_time -= a_time[0]
    a_power = np.array([r['gpu_power'] if r['gpu_power'] > 0 else r['cpu_power'] for r in adv_readings])
    
    ax.plot(b_time, b_power, 'b-', label='Baseline Power', alpha=0.7)
    ax.plot(a_time, a_power, 'r-', label='Adversarial Power', alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (Watts)')
    ax.set_title(f'DeepAR {attack_mode.title()} Attack - Power Monitoring Trace')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def plot_metric_comparison(metrics, metric_name, model_name, save_path):
    labels = list(metrics.keys())
    values = list(metrics.values())
    
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#4c72b0', '#dd8452']
    bars = ax.bar(labels, values, color=colors, width=0.5)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel(metric_name)
    ax.set_title(f'{model_name}: Baseline vs Adversarial', fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    if values[0] > 0:
        increase = ((values[1] - values[0]) / values[0]) * 100
        ax.text(0.5, 0.9, f'+{increase:.1f}% Increase', 
                transform=ax.transAxes, ha='center', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
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
        metrics = get_chronos_metrics()
        metric_name = "Latency (ms)"
        save_name = "chronos_latency_metric_comparison.png"
        
    elif args.model == "deepar":
        if "bitflip" in args.attack:
            mode = args.attack.split("_")[1]
            metrics = get_deepar_metrics(mode)
            metric_name = "Energy (mJ)"
            save_name = f"deepar_{args.attack}_metric_comparison.png"
        elif args.attack == "energy":
            # Direct Energy attack (Blackbox)
            # Re-use get_deepar_metrics but force load the energy file
            # We need to modify get_deepar_metrics slightly or just handle filenames there
             metrics = get_deepar_metrics("energy")
             metric_name = "Energy (mJ)"
             save_name = "deepar_energy_metric_comparison.png"
            
    plot_metric_comparison(metrics, metric_name, f"{args.model.upper()} {args.attack}", save_name)

if __name__ == "__main__":
    main()

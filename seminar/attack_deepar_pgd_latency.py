import torch
import numpy as np

from config import (CONTEXT_LEN, NUM_FEATURES, BASELINE_REPS, VERIFICATION_REPS,
                    PGD_EPSILON, PGD_ALPHA, PGD_NUM_STEPS, PGD_WARMUP_REPS)
from utils.model_loader import load_deepar, load_seed_from_csv, make_predictor, get_device
from utils.power_monitor import PowerMonitor
from utils.metrics import measure_latency
from utils.pgd import PGDAttack
from utils.attack_runner import print_results
from utils.visualization import plot_pgd_results

print("=" * 70)
print("DeepAR-LSTM PGD LATENCY ATTACK (WHITE-BOX)")
print("=" * 70)

device = get_device()
power_monitor = PowerMonitor(0.01)

model, checkpoint = load_deepar(device=device)
seed_data, mean, std = load_seed_from_csv()
seed_tensor = torch.tensor(seed_data, dtype=torch.float32).unsqueeze(0).to(device)
predict = make_predictor(model, device)


def latency_loss_fn(output, x):
    return output.abs().sum() + (output ** 2).sum() * 0.1


print("\nMeasuring baseline...")
baseline_stats = measure_latency(predict, seed_data, power_monitor, device, num_reps=BASELINE_REPS)
print(f"Baseline Latency: {baseline_stats['latency'] * 1000:.2f}ms")

if __name__ == "__main__":
    print("\nStarting PGD Attack (maximizing hidden state stress)...")

    dummy = torch.randn(1, CONTEXT_LEN, NUM_FEATURES).to(device)
    for _ in range(PGD_WARMUP_REPS):
        model(dummy)

    model.train()

    attack = PGDAttack(
        model=model, loss_fn=latency_loss_fn,
        epsilon=PGD_EPSILON, alpha=PGD_ALPHA, num_steps=PGD_NUM_STEPS,
        device=device, maximize=True
    )

    step_data = {'step': [], 'loss': [], 'best_loss': []}

    def on_step(step, loss, best_loss):
        step_data['step'].append(step)
        step_data['loss'].append(loss)
        step_data['best_loss'].append(best_loss)
        if step % 10 == 0:
            print(f"Step {step:3d}: OutputMag={loss:.2f} (Best={best_loss:.2f})")

    adv_tensor, _ = attack.attack(seed_tensor, callback=on_step)

    model.eval()
    adv_input = adv_tensor.squeeze(0).cpu().numpy()

    adv_stats = measure_latency(predict, adv_input, power_monitor, device, VERIFICATION_REPS)
    base_stats = measure_latency(predict, seed_data, power_monitor, device, VERIFICATION_REPS)

    with torch.no_grad():
        base_mag = model(seed_tensor).abs().sum().item()
        adv_mag = model(adv_tensor).abs().sum().item()

    print_results("DeepAR PGD LATENCY ATTACK", [
        ("Output Magnitude", base_mag, adv_mag, "{:.2f}"),
        ("Latency (ms)", base_stats['latency'] * 1000, adv_stats['latency'] * 1000, "{:.2f}"),
    ])

    np.save("deepar_pgd_latency_best_input.npy", adv_input)

    plot_pgd_results(step_data, base_stats, adv_stats,
                     [('Output Mag', base_mag, adv_mag),
                      ('Latency (ms)', base_stats['latency'] * 1000, adv_stats['latency'] * 1000)],
                     'DeepAR-LSTM PGD Latency Attack', 'deepar_pgd_latency_results.png')

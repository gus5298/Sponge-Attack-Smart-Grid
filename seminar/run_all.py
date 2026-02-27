#!/usr/bin/env python3
"""Run the full pipeline: train models, run all attacks, analysis, and visualization."""

import subprocess
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "outputs", "models")

# ─── Pipeline stages ──────────────────────────────────────────────────────────

TRAIN = [
    ("Train DeepAR-LSTM",  [sys.executable, "training/train_deepar.py"],  "deepar_model.pt"),
    ("Train ACT-LSTM",     [sys.executable, "training/train_act.py"],     "act_model.pt"),
]

ATTACKS = [
    ("DeepAR GA Latency",      [sys.executable, "attacks/deepar/ga_latency.py",  "--generations", "20", "--mode", "extreme"]),
    ("DeepAR GA Energy",       [sys.executable, "attacks/deepar/ga_energy.py",   "--generations", "20", "--mode", "extreme"]),
    ("DeepAR PGD Latency",     [sys.executable, "attacks/deepar/pgd_latency.py"]),
    ("DeepAR PGD Energy",      [sys.executable, "attacks/deepar/pgd_energy.py"]),
    ("DeepAR Bitflip",         [sys.executable, "attacks/deepar/bitflip.py"]),
    ("ACT GA Latency",         [sys.executable, "attacks/act/ga_latency.py",     "--generations", "20", "--mode", "extreme"]),
    ("ACT GA Energy",          [sys.executable, "attacks/act/ga_energy.py",      "--generations", "20", "--mode", "extreme"]),
    ("ACT PGD Latency",        [sys.executable, "attacks/act/pgd_latency.py"]),
    ("ACT PGD Energy",         [sys.executable, "attacks/act/pgd_energy.py"]),
    ("ACT Bitflip",            [sys.executable, "attacks/act/bitflip.py"]),
    ("Chronos GA Latency",     [sys.executable, "attacks/chronos/ga_latency.py", "--generations", "20", "--mode", "extreme"]),
    ("Chronos GA Energy",      [sys.executable, "attacks/chronos/ga_energy.py",  "--mode", "extreme", "--generations", "20"]),
    ("Chronos PGD Latency",    [sys.executable, "attacks/chronos/pgd_latency.py"]),
    ("Chronos PGD Energy",     [sys.executable, "attacks/chronos/pgd_energy.py"]),
    ("Chronos Bitflip",        [sys.executable, "attacks/chronos/bitflip.py"]),
]

ANALYSIS = [
    ("Compare Latency",        [sys.executable, "analysis/compare_all_attacks.py",        "--metric", "latency"]),
    ("Compare Energy",         [sys.executable, "analysis/compare_all_attacks.py",        "--metric", "energy"]),
    ("Energy Power Compare",   [sys.executable, "analysis/compare_energy_power.py"]),
    ("XAI SHAP",               [sys.executable, "analysis/xai_shap_improved.py"]),
    ("XAI Advanced",           [sys.executable, "analysis/xai_advanced_analysis.py"]),
    ("XAI PGD",                [sys.executable, "analysis/xai_pgd_analysis.py"]),
    ("XAI Ponder",             [sys.executable, "analysis/xai_ponder_analysis.py"]),
    ("Adversarial Patterns",   [sys.executable, "analysis/analyze_adversarial_patterns.py"]),
]

VISUALIZATION = [
    ("Unified Diagrams",       [sys.executable, "visualization/generate_unified_diagrams.py"]),
    ("XAI Diagrams",           [sys.executable, "visualization/generate_xai_diagrams.py"]),
    ("Metric Diagrams",        [sys.executable, "visualization/generate_metric_diagrams.py"]),
    ("Optimization History",   [sys.executable, "visualization/generate_optimization_history.py"]),
    ("Heatmaps PDF (latency)", [sys.executable, "visualization/export_heatmaps_pdf.py", "--metric", "latency"]),
    ("Heatmaps PDF (energy)",  [sys.executable, "visualization/export_heatmaps_pdf.py", "--metric", "energy"]),
]


def fmt_time(secs):
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs/60:.1f}m"
    return f"{secs/3600:.1f}h"


def run_task(desc, cmd):
    print(f"  > {' '.join(cmd)}")
    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=BASE_DIR)
        elapsed = time.time() - t0
        ok = result.returncode == 0
        status = "OK" if ok else f"FAILED (exit {result.returncode})"
    except Exception as e:
        elapsed = time.time() - t0
        ok = False
        status = f"ERROR: {e}"

    print(f"  {status} [{fmt_time(elapsed)}]\n")
    return desc, ok, elapsed


def run_training(tasks):
    print(f"\n{'=' * 70}")
    print(f"  STAGE: TRAINING  ({len(tasks)} tasks)")
    print(f"{'=' * 70}\n")

    results = []
    for i, (desc, cmd, weight_file) in enumerate(tasks, 1):
        weight_path = os.path.join(MODELS_DIR, weight_file)
        if os.path.exists(weight_path):
            print(f"[{i}/{len(tasks)}] {desc} — SKIPPED (weights exist: {weight_file})\n")
            results.append((desc, True, 0))
            continue
        print(f"[{i}/{len(tasks)}] {desc}")
        results.append(run_task(desc, cmd))

    return results


def run_stage(stage_name, tasks):
    print(f"\n{'=' * 70}")
    print(f"  STAGE: {stage_name}  ({len(tasks)} tasks)")
    print(f"{'=' * 70}\n")

    results = []
    for i, (desc, cmd) in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {desc}")
        results.append(run_task(desc, cmd))

    return results


def main():
    print("=" * 70)
    print("  FULL PIPELINE RUN")
    print("=" * 70)

    all_results = []
    t_total = time.time()

    # Training (skip if weights exist)
    train_results = run_training(TRAIN)
    all_results.extend(train_results)

    # Check weights exist before proceeding
    missing = [f for f in ["deepar_model.pt", "act_model.pt"]
               if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        print(f"\n*** Missing model weights: {', '.join(missing)} ***")
        print(f"*** Cannot proceed with attacks. ***")
    else:
        for stage_name, tasks in [("ATTACKS", ATTACKS), ("ANALYSIS", ANALYSIS), ("VISUALIZATION", VISUALIZATION)]:
            all_results.extend(run_stage(stage_name, tasks))

    elapsed_total = time.time() - t_total

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY  (total: {fmt_time(elapsed_total)})")
    print(f"{'=' * 70}")
    for desc, ok, elapsed in all_results:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {desc:30s}  {fmt_time(elapsed):>8s}")

    passed = sum(1 for _, ok, _ in all_results if ok)
    failed = sum(1 for _, ok, _ in all_results if not ok)
    print(f"\n  {passed} passed, {failed} failed out of {len(all_results)} tasks")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

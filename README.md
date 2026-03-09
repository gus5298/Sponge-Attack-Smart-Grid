# Sponge Attacks on Time-Series Forecasting Models

Adversarial "sponge attack" research on wind power forecasting models. Instead of degrading prediction accuracy, these attacks craft inputs that maximize inference **latency**, **energy consumption**, or **hardware stress** (bit-flip transitions).

## Target Models

| Model | Type | Attack Surface |
|-------|------|----------------|
| **DeepAR-LSTM** | Standard LSTM (4 layers, 512 hidden) | Fixed compute per input |
| **ACT-LSTM** | LSTM with Adaptive Computation Time | Variable ponder steps (up to 20) per time step |
| **Chronos** | Pretrained transformer ([amazon/chronos-t5-small](https://huggingface.co/amazon/chronos-t5-small)) | Tokenizer + autoregressive decode |

## Attack Methods

| Method | Type | Applicable Models | Fitness / Loss |
|--------|------|-------------------|----------------|
| Genetic Algorithm | Black-box | All three | Real measured latency or energy |
| PGD (Projected Gradient Descent) | White-box | DeepAR, ACT-LSTM | Output magnitude sum |
| Bit-Flip Oracle | Black-box | DeepAR | Bit transitions (hardware stress proxy) |
| Ponder Maximization | Black-box | ACT-LSTM | Ponder steps squared |

The GA-based attacks use `pygad` with custom time-slice crossover and model-specific mutation operators. White-box PGD attacks require gradient access and are not possible for Chronos (non-differentiable pipeline).

## Setup

### Requirements

Python 3.10+. Install dependencies:

```bash
pip install torch pygad psutil matplotlib pandas numpy scipy transformers accelerate chronos-forecasting pypower
```

### Docker

```bash
docker build -t src src/
docker run src                                    # trains DeepAR by default
docker run src python attack_deepar_latency.py    # run a specific script
```

## Usage

All commands are run from the `src/` directory.

### 1. Train Models

Models must be trained before running attacks:

```bash
python train_deepar.py       # -> deepar_model.pt
python train_act.py          # -> act_model.pt
```

Chronos uses a pretrained HuggingFace checkpoint and requires no training.

### 2. Run Attacks

Each attack script accepts CLI arguments for `--generations` and `--mode` (constrained or extreme):

```bash
# DeepAR attacks
python attack_deepar_latency.py --generations 50 --mode extreme
python attack_deepar_energy.py --generations 50 --mode extreme
python attack_deepar_bitflip.py --generations 30 --mode blackbox
python attack_deepar_bitflip.py --generations 30 --mode whitebox
python attack_deepar_pgd_latency.py
python attack_deepar_pgd_energy.py

# ACT-LSTM attacks
python attack_act_latency.py
python attack_act_energy.py
python attack_act_full.py
python attack_act_pgd_latency.py
python attack_act_pgd_energy.py

# Chronos attacks
python attack_chronos_latency.py --generations 50 --mode extreme
python attack_energy_sponge.py --model chronos --generations 50
```

**Mode descriptions:**
- `constrained` — Population initialized near the seed data with small perturbations
- `extreme` — Population includes denormalized floats, extreme magnitudes, and adversarial patterns

### 3. Analysis and Visualization

```bash
# Cross-model comparisons
python compare_all_attacks.py --metric latency   # or --metric energy
python compare_energy_power.py

# Diagram generation
python generate_unified_diagrams.py
python generate_xai_diagrams.py
python generate_metric_diagrams.py
python generate_optimization_history.py

# XAI / explainability analysis
python xai_shap_improved.py
python xai_advanced_analysis.py
python xai_pgd_analysis.py
python xai_ponder_analysis.py
python analyze_adversarial_patterns.py

# PDF exports
python export_heatmaps_pdf.py --metric latency   # or --metric energy
```

## Data

Wind turbine measurements from 4 locations (`data/Location{1-4}.csv`), each containing hourly readings of:

| Column | Description |
|--------|-------------|
| `Power` | Turbine output, normalized 0-1 |
| `temperature_2m` | Temperature (F) at 2m |
| `relativehumidity_2m` | Relative humidity (%) at 2m |
| `dewpoint_2m` | Dew point (F) at 2m |
| `windspeed_10m` | Wind speed (m/s) at 10m |
| `windspeed_100m` | Wind speed (m/s) at 100m |
| `winddirection_10m` | Wind direction (degrees) at 10m |
| `winddirection_100m` | Wind direction (degrees) at 100m |
| `windgusts_10m` | Wind gusts (m/s) at 10m |

Models consume sliding windows of 64 time steps across all 9 features and predict the next 10 Power values.

## Output Artifacts

Attack scripts produce:
- `*_best_input.npy` — Best adversarial input found
- `*_generation_data.npz` — Per-generation fitness, latency, power metrics
- `*_hof_*.npy` — Hall-of-fame top-10 solutions
- `*_results.png` — Evolution plots (fitness, latency, power, CPU%)

## Project Structure

```
src/
  config.py                  # Central hyperparameters and constants
  models/
    deepar.py                # DeepARLSTM model
    act.py                   # ACTLSTMCell + ACTModel (Adaptive Computation Time)
  utils/
    data_loader.py           # CSV loading and normalization
    dataset.py               # TimeSeriesDataset (sliding window)
    metrics.py               # measure_latency(), measure_energy()
    power_monitor.py         # Threaded CPU/GPU power sampling via psutil/nvidia-smi
    pgd.py                   # PGDAttack (white-box gradient attack)
    bitflip_oracle.py        # Bit-flip counting for hardware stress estimation
    model_loader.py          # Shared model/seed loading (load_deepar, load_act, load_seed, make_predictor)
    ga_operators.py          # Shared GA crossover and mutation operators
    attack_runner.py         # Common GA attack runner logic
    visualization.py         # Shared plotting functions for attack results
    xai_helpers.py           # Shared XAI utilities (adversarial input loading, energy/latency wrappers)
  train_deepar.py            # DeepAR training script
  train_act.py               # ACT-LSTM training script
  attack_*.py                # Attack scripts (GA and PGD variants)
  compare_*.py               # Cross-model comparison plots
  generate_*.py              # Diagram and figure generation
  xai_*.py                   # Explainability analysis scripts
  export_heatmaps_pdf.py     # PDF heatmap exports (--metric latency|energy)
  data/
    Location{1-4}.csv        # Wind turbine datasets
  Dockerfile                 # Python 3.10 container with all deps
```

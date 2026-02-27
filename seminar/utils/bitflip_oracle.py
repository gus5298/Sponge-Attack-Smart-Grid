"""
BitFlipOracle: Software-based bit-flip estimation for sponge attacks.

This module measures "hardware stress" by counting bit-level differences
between the input data and the model's first-layer weights (whitebox).
The input is tiled to match the full weight matrix size, so flip counts
scale with model size.
"""
import torch
import numpy as np


def _popcount32(x: np.ndarray) -> np.ndarray:
    """Count set bits in 32-bit integers (population count)."""
    x = x.astype(np.uint32)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0f0f0f0f
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x3f


class BitFlipOracle:
    """
    Measures simulated 'hardware stress' via bit-flip counts.

    Compares input bits against first-layer weight bits. The input is
    tiled (repeated) to cover the full weight matrix, so larger models
    produce proportionally higher flip counts.
    """

    def __init__(self, model):
        """
        Initialize the oracle.

        Args:
            model: PyTorch model (required). First layer weights are cached.
        """
        if model is None:
            raise ValueError("BitFlipOracle requires a model.")
        self.model = model
        self._first_layer_weights_bits = None
        self._cache_first_layer_weights()

    def _cache_first_layer_weights(self):
        """Cache the bit representation of the first layer's weights."""
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                weights_np = param.detach().cpu().numpy().flatten().astype(np.float32)
                self._first_layer_weights_bits = weights_np.view(np.int32)
                print(f"[BitFlipOracle] Cached weights from '{name}' "
                      f"(shape: {param.shape}, {len(self._first_layer_weights_bits)} elements)")
                break

    def count_flips(self, input_array: np.ndarray) -> int:
        """
        Count bit flips between input and first-layer weights.

        The input is tiled to match the full weight vector length,
        so flip count scales with model size.

        Args:
            input_array: Shape (seq_len, num_features) or (batch, seq_len, num_features).

        Returns:
            Total number of bit differences.
        """
        if self._first_layer_weights_bits is None:
            raise RuntimeError("No weights cached. First layer not found in model.")

        if input_array.ndim == 3:
            input_array = input_array[0]

        flat = input_array.flatten().astype(np.float32)
        input_bits = flat.view(np.int32)

        # Tile input to match full weight matrix size
        weight_len = len(self._first_layer_weights_bits)
        reps = int(np.ceil(weight_len / len(input_bits)))
        tiled = np.tile(input_bits, reps)[:weight_len]

        xor_result = tiled ^ self._first_layer_weights_bits
        flip_counts = _popcount32(xor_result)
        return int(flip_counts.sum())

    def get_flip_ratio(self, input_array: np.ndarray, baseline_array: np.ndarray) -> float:
        """
        Get the ratio of flips compared to a baseline input.

        Args:
            input_array: Adversarial input.
            baseline_array: Original/clean input.

        Returns:
            Ratio of adversarial flips / baseline flips.
        """
        adv_flips = self.count_flips(input_array)
        base_flips = self.count_flips(baseline_array)
        return adv_flips / base_flips if base_flips > 0 else 1.0


def run_sanity_check():
    """Quick sanity check with a dummy model."""
    import torch.nn as nn

    print("=" * 50)
    print("BitFlipOracle Sanity Check")
    print("=" * 50)

    # Create a small dummy model
    dummy_model = nn.Linear(10, 20)

    oracle = BitFlipOracle(model=dummy_model)

    # Test: Random noise
    noise = np.random.randn(100).astype(np.float32)
    flips_noise = oracle.count_flips(noise.reshape(100, 1))
    print(f"Flips in random noise (tiled to weights): {flips_noise}")

    # Test: All zeros
    zeros = np.zeros(100, dtype=np.float32)
    flips_zeros = oracle.count_flips(zeros.reshape(100, 1))
    print(f"Flips in zeros: {flips_zeros}")

    print("=" * 50)
    print("Sanity check complete!")


if __name__ == "__main__":
    run_sanity_check()

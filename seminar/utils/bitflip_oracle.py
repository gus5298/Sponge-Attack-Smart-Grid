"""
BitFlipOracle: Software-based bit-flip estimation for sponge attacks.

This module provides a differentiability-agnostic way to measure "hardware stress"
by counting the number of bit-level transitions in the input data.

Modes:
    - blackbox: Counts transitions within the input tensor (step t vs t-1).
    - whitebox: Counts bit differences between input and model weights (requires model access).
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

    This is used as a fitness metric for genetic algorithms attacking
    models where standard power sensors are too noisy.
    """

    def __init__(self, mode: str = "blackbox", model=None):
        """
        Initialize the oracle.

        Args:
            mode: "blackbox" (input self-transitions) or "whitebox" (input vs weights).
            model: PyTorch model (required for whitebox mode).
        """
        if mode not in ("blackbox", "whitebox"):
            raise ValueError(f"Unknown mode: {mode}. Use 'blackbox' or 'whitebox'.")
        self.mode = mode
        self.model = model
        self._first_layer_weights_bits = None

        if mode == "whitebox":
            if model is None:
                raise ValueError("Whitebox mode requires a model.")
            self._cache_first_layer_weights()

    def _cache_first_layer_weights(self):
        """Cache the bit representation of the first layer's weights."""
        # Find the first layer with weights
        for name, param in self.model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                weights_np = param.detach().cpu().numpy().flatten().astype(np.float32)
                self._first_layer_weights_bits = weights_np.view(np.int32)
                print(f"[BitFlipOracle] Cached weights from '{name}' (shape: {param.shape})")
                break

    def count_flips_blackbox(self, input_array: np.ndarray) -> int:
        """
        Count bit flips between consecutive time steps in the input.

        This simulates the transitions in the data bus as sequential
        values are loaded into registers.

        Args:
            input_array: Shape (seq_len, num_features) or (batch, seq_len, num_features).

        Returns:
            Total number of bit flips across all transitions.
        """
        if input_array.ndim == 3:
            input_array = input_array[0]  # Take first batch item

        flat = input_array.flatten().astype(np.float32)
        bits = flat.view(np.int32)

        # XOR consecutive elements to find differing bits
        xor_result = bits[:-1] ^ bits[1:]

        # Count set bits in XOR result
        flip_counts = _popcount32(xor_result)
        return int(flip_counts.sum())

    def count_flips_whitebox(self, input_array: np.ndarray) -> int:
        """
        Count bit flips between input and first-layer weights.

        This represents the 'activity' in the first matrix multiplication.

        Args:
            input_array: Shape (seq_len, num_features) or (batch, seq_len, num_features).

        Returns:
            Total number of bit differences.
        """
        if self._first_layer_weights_bits is None:
            raise RuntimeError("No weights cached. Call _cache_first_layer_weights first.")

        if input_array.ndim == 3:
            input_array = input_array[0]

        flat = input_array.flatten().astype(np.float32)
        input_bits = flat.view(np.int32)

        # Broadcast: compare each input element to subset of weights
        min_len = min(len(input_bits), len(self._first_layer_weights_bits))
        xor_result = input_bits[:min_len] ^ self._first_layer_weights_bits[:min_len]

        flip_counts = _popcount32(xor_result)
        return int(flip_counts.sum())

    def count_flips(self, input_array: np.ndarray) -> int:
        """
        Count bit flips based on the configured mode.

        Args:
            input_array: Input data.

        Returns:
            Bit-flip count.
        """
        if self.mode == "blackbox":
            return self.count_flips_blackbox(input_array)
        else:
            return self.count_flips_whitebox(input_array)

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
    """Quick sanity check to verify popcount logic."""
    print("=" * 50)
    print("BitFlipOracle Sanity Check")
    print("=" * 50)

    # Test 1: All zeros vs all ones
    zeros = np.zeros(10, dtype=np.float32)
    ones = np.ones(10, dtype=np.float32)

    oracle = BitFlipOracle(mode="blackbox")

    flips_zeros = oracle.count_flips(zeros.reshape(10, 1))
    flips_ones = oracle.count_flips(ones.reshape(10, 1))

    print(f"Flips in constant zeros: {flips_zeros} (expected: 0)")
    print(f"Flips in constant ones: {flips_ones} (expected: 0)")

    # Test 2: Alternating pattern
    alternating = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    flips_alt = oracle.count_flips(alternating.reshape(10, 1))
    print(f"Flips in 0-1 alternating: {flips_alt} (expected: high)")

    # Test 3: Random noise
    noise = np.random.randn(100).astype(np.float32)
    flips_noise = oracle.count_flips(noise.reshape(100, 1))
    print(f"Flips in random noise: {flips_noise}")

    print("=" * 50)
    print("Sanity check complete!")


if __name__ == "__main__":
    run_sanity_check()

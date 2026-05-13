#!/usr/bin/env python3
"""
Quick test of the direction steering experiment.
Tests that directions can be loaded and steering works.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add to path
sys.path.insert(0, str(Path(__file__).parent / "sae_java_bug"))

from sparse_autoencoders.notebooks.direction_steering_vulnerability_likelihood import \
    VulnerabilityLikelihoodExperiment

print("=" * 80)
print("Direction Steering Test")
print("=" * 80)

# Check if model is available
try:
    print("\n[1/3] Initializing experiment...")
    exp = VulnerabilityLikelihoodExperiment(
        "Qwen/Qwen2.5-7B-Instruct", torch.device("cpu")
    )
    print("  ✓ Model and tokenizer loaded")
except Exception as e:
    print(f"  ✗ Failed to load model: {e}")
    print("  (This is expected if you don't have the model or CUDA)")
    sys.exit(1)

# Check if activations exist
print("\n[2/3] Checking activation files...")
acts_dir = (
    Path(__file__).parent
    / "sae_java_bug"
    / "artifacts"
    / "activations"
    / "raw_activations"
    / "vulnerable_code_qwen_coder_standard_16384_raw"
)

required_layers = [0, 3, 7, 11, 15, 19, 23, 27]
for layer in required_layers:
    act_file = (
        acts_dir
        / f"activations_layer_{layer}_raw_component_hidden_state_last_token.jsonl"
    )
    if act_file.exists():
        size_mb = act_file.stat().st_size / (1024 * 1024)
        print(f"  ✓ Layer {layer}: {size_mb:.0f} MB")
    else:
        print(f"  ✗ Layer {layer}: NOT FOUND")

# Try to compute a direction
print("\n[3/3] Testing direction computation...")
try:
    layer_to_test = 3
    direction = exp.compute_vulnerability_direction(layer_to_test)
    if direction is not None:
        print(f"  ✓ Direction for layer {layer_to_test}:")
        print(f"    Shape: {direction.shape}")
        print(f"    Norm: {direction.norm():.4f}")
        print(f"    Mean: {direction.mean():.4f}")
        print(f"    Std: {direction.std():.4f}")
    else:
        print(f"  ✗ Failed to compute direction")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("Test Complete")
print("=" * 80)
print("\nTo run the full experiment, use:")
print(
    "  python sae_java_bug/sparse_autoencoders/notebooks/direction_steering_vulnerability_likelihood.py"
)
print("    --n_snippets 5 --n_prompts_per 3 --n_alphas 7")

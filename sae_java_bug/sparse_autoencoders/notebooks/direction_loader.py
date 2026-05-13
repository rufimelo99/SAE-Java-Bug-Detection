"""
Direction Loader Utility

Loads pre-computed vulnerability direction vectors for causal intervention experiments.

The vulnerability direction is: d[L] = normalize(mean_vulnerable[L] - mean_secure[L])
Computed across C-code pairs to capture the linear axis separating vulnerable from secure.

Usage:
    from direction_loader import DirectionLoader
    loader = DirectionLoader()
    d_layer = loader.load_direction(layer=11)  # Returns: (D, ) tensor
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch


class DirectionLoader:
    """Load vulnerability direction vectors from disk cache."""

    def __init__(self, repo_root: Optional[Path] = None):
        """
        Initialize direction loader.

        Args:
            repo_root: Path to SAE-Java-Bug-Detection repo root.
                      If None, infers from this file's location.
        """
        if repo_root is None:
            repo_root = Path(__file__).parents[3]

        self.repo_root = repo_root
        self.artifacts_dir = repo_root / "sae_java_bug" / "artifacts"
        self.direction_cache = self.artifacts_dir / "direction_cache"
        self.direction_cache.mkdir(parents=True, exist_ok=True)

        self._cache = {}
        self.layers = [0, 3, 7, 11, 15, 19, 23, 27]

    def compute_directions_from_activations(
        self,
        activations_file: Path,
        output_file: Path = None,
        layers: list = None,
        language: str = "c",
    ) -> Dict[int, torch.Tensor]:
        """
        Compute vulnerability direction from raw activations.

        Direction = normalize(mean_vulnerable - mean_secure)

        Args:
            activations_file: Path to raw activations JSONL
            output_file: Path to save computed directions (pickle)
            layers: Which layers to compute for [default: all]
            language: Filter by language [default: 'c']

        Returns:
            Dictionary mapping layer -> direction tensor
        """
        import base64
        import json
        from collections import defaultdict

        if layers is None:
            layers = self.layers

        if output_file is None:
            output_file = self.direction_cache / f"directions_{language}_only.pt"

        print(f"Computing directions from {activations_file}")

        # Accumulate activations by vulnerability status
        vulnerable_acts = defaultdict(list)
        secure_acts = defaultdict(list)

        with activations_file.open() as f:
            for line_idx, line in enumerate(f):
                if line_idx % 1000 == 0:
                    print(f"  Processing line {line_idx}...")

                try:
                    record = json.loads(line)
                    if record.get("file_extension") != language:
                        continue

                    # Check which status this is
                    is_vulnerable = record.get("vulnerable", False)

                    # Parse activations per layer
                    acts_by_layer = {}
                    for layer in layers:
                        key = f"activations_layer_{layer}"
                        if key in record:
                            acts_tensor = np.array(record[key])
                            acts_by_layer[layer] = acts_tensor

                    # Accumulate
                    for layer, acts in acts_by_layer.items():
                        if is_vulnerable:
                            vulnerable_acts[layer].append(acts)
                        else:
                            secure_acts[layer].append(acts)

                except Exception as e:
                    if line_idx < 10:
                        print(f"  Skipping line {line_idx}: {e}")

        # Compute mean representations and directions
        directions = {}
        for layer in layers:
            if layer not in vulnerable_acts or layer not in secure_acts:
                print(f"  Layer {layer}: insufficient data")
                continue

            v_acts = np.array(vulnerable_acts[layer])
            s_acts = np.array(secure_acts[layer])

            mean_vulnerable = v_acts.mean(axis=0)
            mean_secure = s_acts.mean(axis=0)

            # Direction: from secure to vulnerable
            direction = mean_vulnerable - mean_secure
            direction_norm = direction / (np.linalg.norm(direction) + 1e-8)

            directions[layer] = torch.from_numpy(direction_norm).float()
            print(
                f"  Layer {layer}: direction computed, shape {directions[layer].shape}"
            )

        # Save
        torch.save(directions, output_file)
        print(f"✓ Directions saved to {output_file}")

        return directions

    def load_direction(
        self, layer: int, language: str = "c", device: str = "cpu"
    ) -> Optional[torch.Tensor]:
        """
        Load pre-computed direction for a layer.

        Args:
            layer: Layer number (0, 3, 7, 11, 15, 19, 23, 27)
            language: Language filter (default: 'c')
            device: Device to load onto ('cpu' or 'cuda')

        Returns:
            Direction tensor of shape (hidden_dim,) or None if not found
        """
        cache_key = (layer, language)
        if cache_key in self._cache:
            return self._cache[cache_key].to(device)

        # Try to load from disk
        cache_file = self.direction_cache / f"directions_{language}_only.pt"
        if cache_file.exists():
            directions = torch.load(cache_file, map_location="cpu")
            if layer in directions:
                direction = directions[layer].to(device)
                self._cache[cache_key] = direction.to("cpu")  # Cache on CPU
                return direction

        return None

    def load_all_directions(
        self, language: str = "c", device: str = "cpu"
    ) -> Dict[int, torch.Tensor]:
        """
        Load all pre-computed directions.

        Args:
            language: Language filter
            device: Device to load onto

        Returns:
            Dictionary mapping layer -> direction tensor
        """
        cache_file = self.direction_cache / f"directions_{language}_only.pt"
        if cache_file.exists():
            directions = torch.load(cache_file, map_location=device)
            return directions

        return {}

    def validate_direction(self, direction: torch.Tensor) -> bool:
        """Check if direction tensor is valid (normalized, no NaN)."""
        if direction is None:
            return False
        if direction.isnan().any():
            return False
        norm = direction.norm().item()
        return 0.9 < norm < 1.1  # Should be normalized (norm ≈ 1)


# Example usage
if __name__ == "__main__":
    loader = DirectionLoader()

    # Try loading existing directions
    print("\nAttempting to load pre-computed directions...")
    d = loader.load_all_directions()
    if d:
        print(f"✓ Loaded {len(d)} direction vectors:")
        for layer in sorted(d.keys()):
            direction = d[layer]
            print(
                f"  Layer {layer:2d}: shape {direction.shape}, norm {direction.norm():.4f}"
            )
    else:
        print("✗ No pre-computed directions found.")
        print("\nTo compute directions, run:")
        print("  python cross_layer_direction_probe.py --language c")

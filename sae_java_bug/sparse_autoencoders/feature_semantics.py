"""
Feature Semantics Mapper for SAE Activations

This script maps SAE feature activations to semantic concepts by:
1. Finding top-activating examples for each feature
2. Extracting code patterns that activate features
3. Optionally using an LLM to generate semantic descriptions
4. Creating a feature → semantic concept mapping

Usage:
    python feature_semantics.py --run_id run_20260128_184854 --top_k 10
"""

import argparse
import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from tqdm import tqdm
import orjson


@dataclass
class FeatureSemantics:
    """Semantic description of an SAE feature."""
    feature_idx: int
    feature_type: str  # "active" or "dormant"
    activation_freq: float

    # Top activating samples
    top_activating_samples: list[dict]  # [{sample_idx, activation, code_snippet, label}]

    # Activation statistics
    mean_activation: float
    max_activation: float
    std_activation: float

    # Semantic description (from LLM or manual)
    semantic_label: Optional[str] = None
    semantic_description: Optional[str] = None

    # Vulnerability association
    vuln_activation_ratio: float = 0.0  # How much more it activates on vulnerable code
    associated_cwes: list[str] = None


@dataclass
class TopActivation:
    """A top-activating sample for a feature."""
    sample_idx: int
    activation_value: float
    label: str  # "secure" or "vulnerable"
    cwe: Optional[str] = None
    code_snippet: Optional[str] = None
    vuln_id: Optional[str] = None


def load_activations(run_id: str, layer: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Load secure and vulnerable activations."""
    base_path = Path(f"../artifacts/activations/{run_id}")

    secure_path = base_path / f"safe_layer_{layer}.pt"
    vuln_path = base_path / f"vulnerable_layer_{layer}.pt"

    secure = torch.load(secure_path)
    vulnerable = torch.load(vuln_path)

    return secure, vulnerable


def load_original_data(run_id: str, layer: int = 0) -> list[dict]:
    """Load original activation data with code snippets."""
    base_path = Path(f"../artifacts/activations/{run_id}")

    # Find the activation file
    activation_files = list(base_path.glob(f"activations_layer_{layer}_*.jsonl"))
    if not activation_files:
        print(f"Warning: No activation file found for layer {layer}")
        return []

    activation_file = activation_files[0]

    data = []
    with open(activation_file, "r") as f:
        for line in f:
            try:
                record = orjson.loads(line)
                data.append(record)
            except orjson.JSONDecodeError:
                continue

    return data


def load_invariants(run_id: str, layer: int = 0) -> list:
    """Load computed feature invariants."""
    path = Path(f"../artifacts/activations/{run_id}/feature_invariants_layer_{layer}.pkl")

    if not path.exists():
        print(f"Warning: Invariants file not found at {path}")
        return []

    with open(path, "rb") as f:
        return pickle.load(f)


def compute_feature_semantics(
    secure_activations: np.ndarray,
    vuln_activations: np.ndarray,
    invariants: list,
    original_data: list,
    top_k: int = 10,
    focus_features: Optional[list[int]] = None,
) -> list[FeatureSemantics]:
    """
    Compute semantic information for each feature.

    Args:
        secure_activations: (n_secure, n_features) array
        vuln_activations: (n_vuln, n_features) array
        invariants: List of FeatureInvariant objects
        original_data: Original data with code snippets
        top_k: Number of top activations to keep per feature
        focus_features: Optional list of feature indices to analyze (default: all)
    """
    n_features = secure_activations.shape[1]

    # Combine activations for analysis
    all_activations = np.vstack([secure_activations, vuln_activations])
    n_secure = len(secure_activations)

    # Create labels
    labels = ["secure"] * n_secure + ["vulnerable"] * len(vuln_activations)

    # Determine which features to analyze
    if focus_features is None:
        focus_features = list(range(n_features))

    semantics_list = []

    for feat_idx in tqdm(focus_features, desc="Analyzing features"):
        feat_activations = all_activations[:, feat_idx]

        # Get invariant info
        inv = invariants[feat_idx] if feat_idx < len(invariants) else None
        feature_type = inv.feature_type.value if inv else "unknown"
        activation_freq = inv.activation_freq if inv else 0.0

        # Find top activating samples
        top_indices = np.argsort(feat_activations)[-top_k:][::-1]

        top_activations = []
        for idx in top_indices:
            activation_value = float(feat_activations[idx])
            if activation_value < 1e-6:
                continue  # Skip zero activations

            label = labels[idx]

            # Get original data if available
            cwe = None
            code_snippet = None
            vuln_id = None

            if idx < len(original_data):
                record = original_data[idx]
                cwe = record.get("cwe")
                vuln_id = record.get("vuln_id")
                # Code snippet would come from the original dataset

            top_activations.append(TopActivation(
                sample_idx=idx,
                activation_value=activation_value,
                label=label,
                cwe=cwe,
                code_snippet=code_snippet,
                vuln_id=vuln_id,
            ))

        # Compute statistics
        nonzero_mask = feat_activations > 1e-6
        if nonzero_mask.sum() > 0:
            nonzero_activations = feat_activations[nonzero_mask]
            mean_activation = float(np.mean(nonzero_activations))
            max_activation = float(np.max(nonzero_activations))
            std_activation = float(np.std(nonzero_activations))
        else:
            mean_activation = 0.0
            max_activation = 0.0
            std_activation = 0.0

        # Compute vulnerability association
        secure_active = (secure_activations[:, feat_idx] > 1e-6).mean()
        vuln_active = (vuln_activations[:, feat_idx] > 1e-6).mean()
        vuln_activation_ratio = vuln_active / (secure_active + 1e-10)

        # Find associated CWEs
        associated_cwes = []
        for ta in top_activations:
            if ta.label == "vulnerable" and ta.cwe and ta.cwe not in associated_cwes:
                associated_cwes.append(ta.cwe)

        semantics = FeatureSemantics(
            feature_idx=feat_idx,
            feature_type=feature_type,
            activation_freq=activation_freq,
            top_activating_samples=[asdict(ta) for ta in top_activations],
            mean_activation=mean_activation,
            max_activation=max_activation,
            std_activation=std_activation,
            vuln_activation_ratio=vuln_activation_ratio,
            associated_cwes=associated_cwes[:5],  # Top 5 CWEs
        )

        semantics_list.append(semantics)

    return semantics_list


def identify_interesting_features(
    secure_activations: np.ndarray,
    vuln_activations: np.ndarray,
    invariants: list,
    top_n: int = 100,
) -> dict[str, list[int]]:
    """
    Identify the most interesting features for semantic analysis.

    Returns dict with:
        - "vuln_specific": Features that activate much more on vulnerable code
        - "secure_specific": Features that activate much more on secure code
        - "high_variance": Active features with high variance
        - "dormant_awakened": Dormant features that activate on vulnerable code
    """
    n_features = secure_activations.shape[1]

    # Compute activation frequencies
    secure_freq = (secure_activations > 1e-6).mean(axis=0)
    vuln_freq = (vuln_activations > 1e-6).mean(axis=0)

    # Compute mean activations (when active)
    secure_mean = np.zeros(n_features)
    vuln_mean = np.zeros(n_features)

    for i in range(n_features):
        secure_active = secure_activations[:, i][secure_activations[:, i] > 1e-6]
        vuln_active = vuln_activations[:, i][vuln_activations[:, i] > 1e-6]

        if len(secure_active) > 0:
            secure_mean[i] = secure_active.mean()
        if len(vuln_active) > 0:
            vuln_mean[i] = vuln_active.mean()

    # 1. Vulnerability-specific: Higher activation on vulnerable code
    vuln_ratio = (vuln_freq + 1e-10) / (secure_freq + 1e-10)
    vuln_specific = np.argsort(vuln_ratio)[-top_n:][::-1].tolist()

    # 2. Secure-specific: Higher activation on secure code
    secure_ratio = (secure_freq + 1e-10) / (vuln_freq + 1e-10)
    secure_specific = np.argsort(secure_ratio)[-top_n:][::-1].tolist()

    # 3. High variance active features
    combined = np.vstack([secure_activations, vuln_activations])
    feature_std = np.std(combined, axis=0)
    active_mask = (secure_freq >= 0.1) | (vuln_freq >= 0.1)
    masked_std = feature_std * active_mask
    high_variance = np.argsort(masked_std)[-top_n:][::-1].tolist()

    # 4. Dormant features awakened by vulnerabilities
    dormant_mask = secure_freq < 0.1
    vuln_awakening = vuln_freq * dormant_mask
    dormant_awakened = np.argsort(vuln_awakening)[-top_n:][::-1].tolist()

    return {
        "vuln_specific": vuln_specific,
        "secure_specific": secure_specific,
        "high_variance": high_variance,
        "dormant_awakened": dormant_awakened,
    }


def generate_semantic_prompt(feature_semantics: FeatureSemantics) -> str:
    """
    Generate a prompt for an LLM to describe the feature semantics.
    """
    prompt = f"""Analyze this SAE (Sparse Autoencoder) feature from a code vulnerability detection model.

Feature Information:
- Feature Index: {feature_semantics.feature_idx}
- Type: {feature_semantics.feature_type}
- Activation Frequency: {feature_semantics.activation_freq*100:.1f}%
- Vulnerability Activation Ratio: {feature_semantics.vuln_activation_ratio:.2f}x
- Associated CWEs: {', '.join(feature_semantics.associated_cwes) if feature_semantics.associated_cwes else 'None'}

Top Activating Code Samples:
"""

    for i, sample in enumerate(feature_semantics.top_activating_samples[:5]):
        prompt += f"\n{i+1}. [{sample['label'].upper()}] Activation: {sample['activation_value']:.3f}"
        if sample.get('cwe'):
            prompt += f" (CWE: {sample['cwe']})"
        if sample.get('code_snippet'):
            prompt += f"\n   Code: {sample['code_snippet'][:200]}..."

    prompt += """

Based on this information, provide:
1. A short semantic label (2-5 words) describing what this feature detects
2. A brief description (1-2 sentences) of the semantic concept
3. Whether this feature is likely useful for vulnerability detection and why

Format your response as:
LABEL: <semantic label>
DESCRIPTION: <description>
USEFUL: <yes/no and brief explanation>
"""

    return prompt


def save_semantics(
    semantics_list: list[FeatureSemantics],
    output_path: Path,
    format: str = "json",
):
    """Save feature semantics to file."""

    if format == "json":
        output = [asdict(s) for s in semantics_list]
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    elif format == "markdown":
        with open(output_path, "w") as f:
            f.write("# SAE Feature Semantics\n\n")

            for s in semantics_list:
                f.write(f"## Feature {s.feature_idx}\n\n")
                f.write(f"- **Type**: {s.feature_type}\n")
                f.write(f"- **Activation Frequency**: {s.activation_freq*100:.1f}%\n")
                f.write(f"- **Vulnerability Ratio**: {s.vuln_activation_ratio:.2f}x\n")

                if s.semantic_label:
                    f.write(f"- **Semantic Label**: {s.semantic_label}\n")
                if s.semantic_description:
                    f.write(f"- **Description**: {s.semantic_description}\n")
                if s.associated_cwes:
                    f.write(f"- **Associated CWEs**: {', '.join(s.associated_cwes)}\n")

                f.write("\n**Top Activating Samples:**\n\n")
                for i, sample in enumerate(s.top_activating_samples[:5]):
                    f.write(f"{i+1}. [{sample['label']}] activation={sample['activation_value']:.3f}")
                    if sample.get('cwe'):
                        f.write(f" ({sample['cwe']})")
                    f.write("\n")

                f.write("\n---\n\n")

    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Map SAE features to semantic concepts")
    parser.add_argument("--run_id", type=str, required=True, help="Run ID for activations")
    parser.add_argument("--layer", type=int, default=0, help="Layer to analyze")
    parser.add_argument("--top_k", type=int, default=10, help="Top activations per feature")
    parser.add_argument("--top_n_features", type=int, default=100, help="Top features per category")
    parser.add_argument("--output_format", choices=["json", "markdown"], default="json")
    parser.add_argument("--all_features", action="store_true", help="Analyze all features (slow)")

    args = parser.parse_args()

    print(f"Loading activations for {args.run_id}...")
    secure, vulnerable = load_activations(args.run_id, args.layer)
    secure_np = secure.numpy()
    vuln_np = vulnerable.numpy()

    print(f"  Secure samples: {len(secure_np)}")
    print(f"  Vulnerable samples: {len(vuln_np)}")
    print(f"  Features: {secure_np.shape[1]}")

    print(f"\nLoading invariants...")
    invariants = load_invariants(args.run_id, args.layer)
    print(f"  Loaded {len(invariants)} feature invariants")

    print(f"\nLoading original data...")
    original_data = load_original_data(args.run_id, args.layer)
    print(f"  Loaded {len(original_data)} records")

    print(f"\nIdentifying interesting features...")
    interesting = identify_interesting_features(
        secure_np, vuln_np, invariants, top_n=args.top_n_features
    )

    for category, features in interesting.items():
        print(f"  {category}: {len(features)} features")

    # Determine which features to analyze
    if args.all_features:
        focus_features = None
        print(f"\nAnalyzing ALL {secure_np.shape[1]} features...")
    else:
        # Combine interesting features (deduplicated)
        focus_features = list(set(
            interesting["vuln_specific"][:50] +
            interesting["dormant_awakened"][:50] +
            interesting["high_variance"][:25]
        ))
        print(f"\nAnalyzing {len(focus_features)} interesting features...")

    print(f"\nComputing feature semantics...")
    semantics = compute_feature_semantics(
        secure_np, vuln_np, invariants, original_data,
        top_k=args.top_k,
        focus_features=focus_features,
    )

    # Save results
    output_dir = Path(f"../artifacts/activations/{args.run_id}")
    output_path = output_dir / f"feature_semantics_layer_{args.layer}.{args.output_format}"

    save_semantics(semantics, output_path, format=args.output_format)

    # Print summary of most interesting features
    print(f"\n{'='*70}")
    print("Top 10 Vulnerability-Specific Features")
    print(f"{'='*70}")

    vuln_sorted = sorted(semantics, key=lambda s: s.vuln_activation_ratio, reverse=True)
    for s in vuln_sorted[:10]:
        print(f"\nFeature {s.feature_idx}:")
        print(f"  Type: {s.feature_type}")
        print(f"  Vuln ratio: {s.vuln_activation_ratio:.2f}x")
        print(f"  CWEs: {', '.join(s.associated_cwes) if s.associated_cwes else 'None'}")
        if s.semantic_label:
            print(f"  Label: {s.semantic_label}")

    print(f"\n{'='*70}")
    print("Top 10 Dormant Features Awakened by Vulnerabilities")
    print(f"{'='*70}")

    dormant_sorted = sorted(
        [s for s in semantics if s.feature_type == "dormant"],
        key=lambda s: s.vuln_activation_ratio,
        reverse=True
    )
    for s in dormant_sorted[:10]:
        print(f"\nFeature {s.feature_idx}:")
        print(f"  Activation freq: {s.activation_freq*100:.2f}%")
        print(f"  Vuln ratio: {s.vuln_activation_ratio:.2f}x")
        print(f"  CWEs: {', '.join(s.associated_cwes) if s.associated_cwes else 'None'}")


if __name__ == "__main__":
    main()

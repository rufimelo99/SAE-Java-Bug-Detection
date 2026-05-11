"""
Token Position Importance Analysis — Mechanistic Deepening (c)

Analyzes which token positions carry vulnerability signal by computing
per-token projections onto the vulnerability direction.

Key insight: If vulnerability signal is distributed, which positions matter?

Generates:
1. Heatmap: per-position projection magnitude across layers L3-L23
2. Bar chart: position importance (collapsed across layers)
3. Alignment heatmap: % of pairs with positive alignment per position
4. Attention heatmap: how does final token attend to different positions?

Usage:
    conda run -n sae python token_position_importance.py

Saves:
    artifacts/analysis/token_position/
        - position_importance_{layer}.json
        - figures/
            - token_position_importance_heatmap.pdf
            - token_position_alignment_heatmap.pdf
            - final_token_attention_heatmap.pdf
            - position_importance_summary.json
"""

import base64
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
ARTIFACTS = Path(__file__).parents[2] / "artifacts"
OUTPUT_DIR = ARTIFACTS / "analysis" / "token_position"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

# Raw activations directory (all 8 layers)
RAW_RUN_DIR = (
    ARTIFACTS
    / "activations"
    / "raw_activations"
    / "vulnerable_code_qwen_coder_standard_16384_raw"
)

# Configuration
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = [3, 7, 11, 15, 19, 23]  # Focus on main layers
MAX_TOKENS = 2048
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Style
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────


def load_source_records(jsonl_path: Path):
    """Load (vuln_id, secure_code, vulnerable_code, cwe, file_extension) from JSONL."""
    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            try:
                secure_code = base64.b64decode(r["secure_code"]).decode(
                    "utf-8", errors="replace"
                )
                vuln_code = base64.b64decode(r["vulnerable_code"]).decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                secure_code = r.get("secure_code", "")
                vuln_code = r.get("vulnerable_code", "")
            records.append(
                {
                    "vuln_id": r["vuln_id"],
                    "cwe": r["cwe"],
                    "file_extension": r["file_extension"],
                    "secure_code": secure_code,
                    "vulnerable_code": vuln_code,
                }
            )
    return records


# ──────────────────────────────────────────────────────────────────────────────
# Per-token activation extraction
# ──────────────────────────────────────────────────────────────────────────────


def extract_per_token_activations(
    code_str: str,
    tokenizer,
    model,
    target_layers: list,
    device,
    max_tokens: int = MAX_TOKENS,
) -> Dict[int, np.ndarray]:
    """
    Run code_str through model, return {layer_idx: [seq_len, d_model]} activations.

    Returns per-token activations (not pooled) for position importance analysis.
    """
    inputs = tokenizer(
        code_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, d_model]
    result = {}
    seq_len = inputs["input_ids"].shape[1]

    for layer_idx in target_layers:
        h = outputs.hidden_states[layer_idx + 1]  # [1, seq_len, d_model]
        acts = h[0].float().cpu().numpy()  # [seq_len, d_model]
        result[layer_idx] = acts

    return result, seq_len


def get_vulnerability_direction(layer_idx: int) -> np.ndarray:
    """
    Load pre-computed vulnerability direction for a given layer.
    Direction is normalized mean(vulnerable - secure) across all pairs.

    Must be pre-computed from existing analysis (directional_readout_probe.py, etc.)
    """
    direction_file = OUTPUT_DIR / f"vulnerability_direction_L{layer_idx}.npy"

    if direction_file.exists():
        return np.load(direction_file)
    else:
        # Fallback: compute on first call from saved activations
        # For now, return None to indicate we need to compute it
        return None


def compute_vulnerability_direction(
    safe_activations: np.ndarray,
    vuln_activations: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized vulnerability direction as mean(vuln - safe).

    Args:
        safe_activations: [n_samples, d_model]
        vuln_activations: [n_samples, d_model]

    Returns:
        direction: [d_model] (normalized)
    """
    delta = np.mean(vuln_activations - safe_activations, axis=0)
    norm = np.linalg.norm(delta)
    if norm > 0:
        return delta / norm
    return delta


# ──────────────────────────────────────────────────────────────────────────────
# Token position importance computation
# ──────────────────────────────────────────────────────────────────────────────


def compute_token_position_importance(
    safe_per_token: np.ndarray,
    vuln_per_token: np.ndarray,
    vuln_direction: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-token contribution to vulnerability direction.

    Args:
        safe_per_token: [n_pairs, seq_len, d_model]
        vuln_per_token: [n_pairs, seq_len, d_model]
        vuln_direction: [d_model] (normalized)

    Returns:
        position_projections: [max_seq_len] - mean projection per position
        position_alignment: [max_seq_len] - % of pairs with positive alignment
        position_std: [max_seq_len] - std of projections per position
    """
    n_pairs = len(safe_per_token)

    # Compute per-token difference (vulnerable - secure)
    deltas = vuln_per_token - safe_per_token  # [n_pairs, seq_len, d_model]

    # Project each token's delta onto vulnerability direction
    projections = np.dot(deltas, vuln_direction)  # [n_pairs, seq_len]

    # Aggregate across pairs
    position_projections = np.mean(projections, axis=0)  # [seq_len]
    position_alignment = (projections > 0).mean(axis=0)  # [seq_len]
    position_std = np.std(projections, axis=0)  # [seq_len]

    return position_projections, position_alignment, position_std


def get_final_token_attention(
    code_str: str,
    tokenizer,
    model,
    device,
    max_tokens: int = MAX_TOKENS,
) -> np.ndarray:
    """
    Extract attention weights from final token to all positions (averaged across heads/layers).

    Returns:
        attention_weights: [seq_len] - attention from final token to each position
    """
    inputs = tokenizer(
        code_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_tokens,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    # attention: tuple of (n_layers) tensors, each [1, n_heads, seq_len, seq_len]
    seq_len = inputs["input_ids"].shape[1]

    # Average attention across layers and heads: focus on deep layers
    final_token_attn = []
    for layer_idx in range(len(outputs.attentions)):
        attn = outputs.attentions[layer_idx]  # [1, n_heads, seq_len, seq_len]
        # Get attention from final token (index -1) to all positions
        final_token_attn.append(attn[0, :, -1, :].cpu().numpy())  # [n_heads, seq_len]

    # Average: [seq_len]
    final_token_attn = np.mean(np.array(final_token_attn), axis=(0, 1))

    return final_token_attn


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────


def plot_token_importance_heatmap(
    position_projections_by_layer: Dict[int, np.ndarray],
    position_std_by_layer: Dict[int, np.ndarray] = None,
    layer_indices: list = None,
    title: str = "Token Position Importance Across Layers",
    output_path: Path = None,
):
    """Visualize per-token projection magnitude across layers."""
    if layer_indices is None:
        layer_indices = sorted(position_projections_by_layer.keys())

    # Pad sequences to same length (take max)
    max_len = max(len(position_projections_by_layer[l]) for l in layer_indices)

    n_layers = len(layer_indices)
    heatmap_data = np.zeros((n_layers, max_len))

    for i, layer_idx in enumerate(layer_indices):
        proj = position_projections_by_layer[layer_idx]
        heatmap_data[i, : len(proj)] = proj

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    # Heatmap across layers
    vmax = np.abs(heatmap_data).max()
    im1 = ax1.imshow(heatmap_data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax1.set_xlabel("Token Position")
    ax1.set_ylabel("Layer")
    ax1.set_yticks(range(n_layers))
    ax1.set_yticklabels([f"L{layer_indices[i]}" for i in range(n_layers)])
    ax1.set_title(
        "Mean Vulnerability Direction Projection per Token Position (Across Layers)"
    )
    plt.colorbar(im1, ax=ax1, label="Projection Magnitude")

    # Collapsed across layers
    mean_proj = np.nanmean(heatmap_data, axis=0)
    positions = np.arange(len(mean_proj))

    ax2.bar(
        positions,
        mean_proj,
        color="steelblue",
        alpha=0.7,
        edgecolor="navy",
        linewidth=0.5,
    )
    ax2.set_xlabel("Token Position")
    ax2.set_ylabel("Mean Projection")
    ax2.set_title("Average Vulnerability Signal per Position (Collapsed Across L3-L23)")
    ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=150)
        print(f"Saved: {output_path}")

    return fig


def plot_token_alignment_heatmap(
    position_alignment_by_layer: Dict[int, np.ndarray],
    layer_indices: list = None,
    output_path: Path = None,
):
    """Visualize % of pairs with positive alignment per position per layer."""
    if layer_indices is None:
        layer_indices = sorted(position_alignment_by_layer.keys())

    max_len = max(len(position_alignment_by_layer[l]) for l in layer_indices)

    n_layers = len(layer_indices)
    alignment_data = np.zeros((n_layers, max_len))

    for i, layer_idx in enumerate(layer_indices):
        align = position_alignment_by_layer[layer_idx]
        alignment_data[i, : len(align)] = align

    fig, ax = plt.subplots(figsize=(16, 6))

    im = ax.imshow(alignment_data, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{layer_indices[i]}" for i in range(n_layers)])
    ax.set_title(
        "% of Vulnerable-Secure Pairs with Positive Alignment per Token Position"
    )
    plt.colorbar(im, ax=ax, label="Alignment %")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=150)
        print(f"Saved: {output_path}")

    return fig


def plot_final_token_attention_heatmap(
    final_token_attention_by_layer: Dict[int, np.ndarray],
    layer_indices: list = None,
    output_path: Path = None,
):
    """Visualize final token's attention to each position."""
    if layer_indices is None:
        layer_indices = sorted(final_token_attention_by_layer.keys())

    max_len = max(len(final_token_attention_by_layer[l]) for l in layer_indices)

    n_layers = len(layer_indices)
    attn_data = np.zeros((n_layers, max_len))

    for i, layer_idx in enumerate(layer_indices):
        attn = final_token_attention_by_layer[layer_idx]
        attn_data[i, : len(attn)] = attn

    fig, ax = plt.subplots(figsize=(16, 6))

    im = ax.imshow(attn_data, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{layer_indices[i]}" for i in range(n_layers)])
    ax.set_title("Final Token Attention to Each Position (Mean Across Heads)")
    plt.colorbar(im, ax=ax, label="Attention Weight")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=150)
        print(f"Saved: {output_path}")

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main analysis
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print("[*] Token Position Importance Analysis")
    print(f"[*] Device: {DEVICE}")
    print(f"[*] Model: {MODEL_ID}")
    print(f"[*] Layers: {LAYERS}")

    # Load model and tokenizer
    print("\n[*] Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype="auto", device_map=DEVICE
    )
    model.eval()

    # Load source records (C language only, to match paper focus)
    print("[*] Loading source records...")
    source_jsonl = (
        RAW_RUN_DIR / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
    )
    records = load_source_records(source_jsonl)

    # Filter to C only
    c_records = [r for r in records if r["file_extension"] == ".c"]
    print(f"[*] Found {len(c_records)} C samples")

    if not c_records:
        print("[!] No C samples found. Exiting.")
        return

    # Extract per-token activations
    print(f"\n[*] Extracting per-token activations for {len(c_records)} samples...")

    per_token_safe = {layer: [] for layer in LAYERS}
    per_token_vuln = {layer: [] for layer in LAYERS}
    final_token_attn = {layer: [] for layer in LAYERS}

    for idx, record in enumerate(c_records):
        if idx % 100 == 0:
            print(f"  [{idx}/{len(c_records)}]")

        try:
            # Secure code activations
            safe_acts, safe_len = extract_per_token_activations(
                record["secure_code"], tokenizer, model, LAYERS, DEVICE
            )

            # Vulnerable code activations
            vuln_acts, vuln_len = extract_per_token_activations(
                record["vulnerable_code"], tokenizer, model, LAYERS, DEVICE
            )

            # Store per-token (pad to max length within this sample pair)
            max_pair_len = max(safe_len, vuln_len)

            for layer in LAYERS:
                # Pad sequences
                safe = safe_acts[layer]
                vuln = vuln_acts[layer]

                if len(safe) < max_pair_len:
                    safe = np.vstack(
                        [safe, np.zeros((max_pair_len - len(safe), safe.shape[1]))]
                    )
                if len(vuln) < max_pair_len:
                    vuln = np.vstack(
                        [vuln, np.zeros((max_pair_len - len(vuln), vuln.shape[1]))]
                    )

                per_token_safe[layer].append(safe)
                per_token_vuln[layer].append(vuln)

        except Exception as e:
            print(f"  [!] Error on sample {idx}: {e}")
            continue

    # Convert to arrays
    print("[*] Stacking activations...")
    for layer in LAYERS:
        per_token_safe[layer] = np.array(per_token_safe[layer])
        per_token_vuln[layer] = np.array(per_token_vuln[layer])
        print(f"  L{layer}: {per_token_safe[layer].shape}")

    # Compute vulnerability direction and position importance
    print("\n[*] Computing vulnerability directions and position importance...")

    position_projections_by_layer = {}
    position_alignment_by_layer = {}
    position_std_by_layer = {}

    for layer in LAYERS:
        print(f"  L{layer}...", end=" ", flush=True)

        # Flatten sequence dimension to compute direction across all positions
        safe_flat = per_token_safe[layer].reshape(-1, per_token_safe[layer].shape[-1])
        vuln_flat = per_token_vuln[layer].reshape(-1, per_token_vuln[layer].shape[-1])

        # Compute vulnerability direction
        vuln_dir = compute_vulnerability_direction(safe_flat, vuln_flat)

        # Compute per-position importance (using padded sequences, unflattened)
        proj, align, std = compute_token_position_importance(
            per_token_safe[layer],
            per_token_vuln[layer],
            vuln_dir,
        )

        position_projections_by_layer[layer] = proj
        position_alignment_by_layer[layer] = align
        position_std_by_layer[layer] = std

        print(f"max_proj={proj.max():.3f}, mean_align={align.mean():.3f}")

    # Save results
    print("\n[*] Saving results...")
    summary = {
        "n_samples": len(c_records),
        "layers": LAYERS,
        "max_seq_len": max(
            max(p.shape[0] for p in position_projections_by_layer.values())
            for _ in [None]
        ),
    }

    with open(OUTPUT_DIR / "position_importance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save per-layer results
    for layer in LAYERS:
        layer_result = {
            "layer": layer,
            "position_projections": position_projections_by_layer[layer].tolist(),
            "position_alignment": position_alignment_by_layer[layer].tolist(),
            "position_std": position_std_by_layer[layer].tolist(),
        }
        with open(OUTPUT_DIR / f"position_importance_L{layer}.json", "w") as f:
            json.dump(layer_result, f, indent=2)

    # Generate visualizations
    print("[*] Generating visualizations...")

    plot_token_importance_heatmap(
        position_projections_by_layer,
        position_std_by_layer,
        LAYERS,
        output_path=OUTPUT_DIR / "figures" / "token_position_importance_heatmap.pdf",
    )

    plot_token_alignment_heatmap(
        position_alignment_by_layer,
        LAYERS,
        output_path=OUTPUT_DIR / "figures" / "token_position_alignment_heatmap.pdf",
    )

    # Copy to paper figures
    import shutil

    for fig_name in [
        "token_position_importance_heatmap.pdf",
        "token_position_alignment_heatmap.pdf",
    ]:
        src = OUTPUT_DIR / "figures" / fig_name
        dst = PAPER_FIGS / fig_name
        if src.exists():
            shutil.copy(src, dst)
            print(f"[*] Copied to paper: {fig_name}")

    print("\n[✓] Token position importance analysis complete!")
    print(f"[*] Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

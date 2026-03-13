"""
causal_patching.py — Activation patching causal experiment.

Tests whether vulnerability information is causally present at specific
residual-stream positions by patching secure activations into vulnerable
forward passes at a target layer, then measuring the shift in the mean-token
representation at a downstream measurement layer.

Design
------
For each (vulnerable, secure) pair and each patch_layer in {3, 7, 11, 15}:
  1. Run the secure sample → cache residual stream at patch_layer.
  2. Run the vulnerable sample with no patch → record baseline mean-token
     projection onto the vulnerability direction at measurement_layer.
  3. Re-run the vulnerable sample with the secure activations injected at
     patch_layer, for three position subsets:
       all       — every shared prefix position
       body      — shared prefix, excluding the last token
       last_only — only the last token position
  4. Record the patched mean-token projection at measurement_layer.

Metric
------
  Δ = proj(patched) − proj(baseline)
  Negative Δ means the representation shifted toward the secure direction.

Key prediction
--------------
  body >> last_only  →  signal is distributed, not concentrated at the
                         final token, confirming the negative-space hypothesis.

Usage
-----
    conda run -n demeanor python causal_patching.py [--n_pairs 200]

Saves
-----
    artifacts/causal_patching/patching_results.json
    <paper_figs>/fig_causal_patching.pdf
"""

import argparse
import base64
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────────
ARTIFACTS  = Path(__file__).parents[2] / "artifacts"
ACT_DIR    = ARTIFACTS / "activations"
OUT_DIR    = ARTIFACTS / "causal_patching"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

SOURCE_JSONL = (
    ACT_DIR / "raw_activations"
    / "vulnerable_code_qwen_coder_standard_16384_raw"
    / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
)

MODEL_ID          = "Qwen/Qwen2.5-7B-Instruct"
PATCH_LAYERS      = [0, 3, 7, 11, 15, 19, 23, 27]  # all 8 sampled layers
MEASUREMENT_LAYER = 27          # must be >= all patch layers; use final sampled layer
MAX_TOKENS        = 512         # shorter for tractable per-pair inference
SEED              = 42
POSITION_SUBSETS  = ["all", "body", "last_only"]

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7, "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ── Data ───────────────────────────────────────────────────────────────────────

def load_pairs(jsonl_path: Path, n_pairs: int | None = None) -> list[dict]:
    records = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            try:
                sec  = base64.b64decode(r["secure_code"]).decode("utf-8", errors="replace")
                vuln = base64.b64decode(r["vulnerable_code"]).decode("utf-8", errors="replace")
            except Exception:
                sec  = r.get("secure_code", "")
                vuln = r.get("vulnerable_code", "")
            if sec.strip() and vuln.strip():
                records.append({
                    "secure": sec, "vuln": vuln,
                    "vuln_id": r.get("vuln_id", ""),
                    "cwe": r.get("cwe", ""),
                })
            if n_pairs and len(records) >= n_pairs:
                break
    return records


# ── Vulnerability direction ────────────────────────────────────────────────────

def load_vuln_direction(layer: int) -> torch.Tensor:
    """Unit-norm vulnerability direction at `layer` from pre-saved mean-pool tensors."""
    for run_dir in sorted((ACT_DIR / "mean_pool").iterdir(), reverse=True):
        sp = run_dir / f"safe_layer_{layer}.pt"
        vp = run_dir / f"vulnerable_layer_{layer}.pt"
        if sp.exists() and vp.exists():
            s = torch.load(sp, map_location="cpu", weights_only=True).float()
            v = torch.load(vp, map_location="cpu", weights_only=True).float()
            return F.normalize((v.mean(0) - s.mean(0)).unsqueeze(0), dim=-1).squeeze(0)
    raise FileNotFoundError(f"No mean_pool tensors for layer {layer}")


# ── Hook utilities ─────────────────────────────────────────────────────────────

def _hidden_from_output(output):
    """
    Extract the hidden-state tensor from a layer output regardless of whether
    the layer returned a plain tensor or a (hidden_states, ...) tuple.
    Newer transformers versions may return a raw tensor instead of a 1-tuple.
    Returns tensor of shape [batch, seq, d_model].
    """
    if isinstance(output, torch.Tensor):
        return output
    return output[0]


def _replace_hidden(output, new_hidden):
    """Return output with the hidden-state component replaced by new_hidden."""
    if isinstance(output, torch.Tensor):
        return new_hidden
    out = list(output)
    out[0] = new_hidden
    return tuple(out)


# ── Hook helpers ───────────────────────────────────────────────────────────────

def make_cache_hook(store: dict, key: str):
    def hook(module, inp, output):
        store[key] = _hidden_from_output(output).detach().clone()  # [1, T, d_model]
    return hook


def make_patch_hook(secure_act: torch.Tensor, subset: str):
    """
    Replaces positions in the vulnerable activation with secure activations.

    secure_act : [1, T_s, d_model]
    subset     : 'all' | 'body' | 'last_only'
    """
    def hook(module, inp, output):
        h   = _hidden_from_output(output).clone()   # [1, T_v, d_model]
        T_v = h.shape[1]
        T_s = secure_act.shape[1]
        src = secure_act.to(device=h.device, dtype=h.dtype)

        if subset == "last_only":
            h[0, -1, :] = src[0, -1, :]
        else:
            T_min = min(T_v, T_s)
            end   = T_min if subset == "all" else max(T_min - 1, 0)
            if end > 0:
                h[0, :end, :] = src[0, :end, :]

        return _replace_hidden(output, h)
    return hook


# ── Single-pair patching ───────────────────────────────────────────────────────

@torch.no_grad()
def cache_all_layers(inputs, model, layers: list[int]) -> dict[int, torch.Tensor]:
    """One forward pass, returns {layer: hidden_state [1, T, d_model]} for all layers."""
    store = {}
    handles = [
        model.model.layers[L].register_forward_hook(make_cache_hook(store, L))
        for L in layers
    ]
    try:
        model(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return store


@torch.no_grad()
def patch_pair(
    vuln_text: str,
    secure_text: str,
    tokenizer,
    model,
    device: torch.device,
    patch_layers: list[int],
    measurement_layer: int,
    vuln_direction: torch.Tensor,
) -> dict[int, dict[str, float]]:
    """
    For each patch_layer, returns projection values (onto vuln_direction at
    measurement_layer) for the baseline and each position subset.

    Optimised: secure activations and the baseline are each extracted in a
    single forward pass; only the 3 patched variants per layer require extra passes.
    Total passes per pair = 2 + len(patch_layers) * len(POSITION_SUBSETS).
    """
    def tokenize(text):
        return tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=MAX_TOKENS,
        ).to(device)

    vuln_inputs   = tokenize(vuln_text)
    secure_inputs = tokenize(secure_text)
    d = vuln_direction.to(device)

    def _mean_proj(h):
        # h : [1, T, d_model]
        return float((h[0].float().mean(0) * d).sum().cpu())

    # 1. ONE pass through secure — cache all patch layers simultaneously
    sec_cache = cache_all_layers(secure_inputs, model, patch_layers)

    # 2. ONE baseline pass through vulnerable — cache measurement layer
    base_cache = cache_all_layers(vuln_inputs, model, [measurement_layer])
    baseline_proj = _mean_proj(base_cache[measurement_layer])

    # 3. For each patch_layer × position_subset: one patched pass
    results = {}
    for patch_layer in patch_layers:
        results[patch_layer] = {"baseline": baseline_proj}
        secure_act = sec_cache[patch_layer]   # [1, T_s, d_model]

        for subset in POSITION_SUBSETS:
            patch_cache = {}
            h_patch = model.model.layers[patch_layer].register_forward_hook(
                make_patch_hook(secure_act, subset)
            )
            h_meas = model.model.layers[measurement_layer].register_forward_hook(
                make_cache_hook(patch_cache, measurement_layer)
            )
            try:
                model(**vuln_inputs, use_cache=False)
            finally:
                h_patch.remove()
                h_meas.remove()

            results[patch_layer][subset] = _mean_proj(patch_cache[measurement_layer])

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=200,
                        help="Number of pairs to process (default 200)")
    args = parser.parse_args()

    print(f"Loading pairs (stratified by file extension)...")
    # Load all then shuffle — the first N records skew toward CWE-79/PHP/JS.
    all_pairs = load_pairs(SOURCE_JSONL, n_pairs=None)
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_pairs)
    pairs = all_pairs[:args.n_pairs]
    print(f"  Using {len(pairs)} pairs (shuffled from {len(all_pairs)} total)")

    print(f"  Loading vulnerability direction (L{MEASUREMENT_LAYER})...")
    vuln_direction = load_vuln_direction(MEASUREMENT_LAYER)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    print(f"  Loading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    # {patch_layer: {subset: [per-pair projection values]}}
    raw = {L: {k: [] for k in ["baseline"] + POSITION_SUBSETS} for L in PATCH_LAYERS}

    for i, pair in enumerate(pairs):
        if i % 25 == 0:
            print(f"  Pair {i + 1}/{len(pairs)}")
        try:
            res = patch_pair(
                pair["vuln"], pair["secure"],
                tokenizer, model, device,
                PATCH_LAYERS, MEASUREMENT_LAYER,
                vuln_direction,
            )
            for patch_layer, layer_res in res.items():
                for key, val in layer_res.items():
                    raw[patch_layer][key].append(val)
        except Exception as e:
            print(f"    [WARN] pair {i}: {e}")

    # Summarise: Δ = patched_proj − baseline_proj
    summary = {}
    print("\nResults (Δ projection; negative = shifted toward secure):")
    for L in PATCH_LAYERS:
        summary[L] = {}
        baselines = np.array(raw[L]["baseline"])
        for subset in POSITION_SUBSETS:
            patched = np.array(raw[L][subset])
            n       = min(len(patched), len(baselines))
            if n == 0:
                continue
            delta = patched[:n] - baselines[:n]
            summary[L][subset] = {
                "mean_delta": float(delta.mean()),
                "sem_delta":  float(delta.std() / np.sqrt(n)),
                "n":          n,
            }
            print(f"  L{L} patch → L{MEASUREMENT_LAYER} meas  "
                  f"{subset:12s}  Δ = {delta.mean():+.4f} ± {delta.std()/np.sqrt(n):.4f}")

    out_json = OUT_DIR / "patching_results.json"
    with out_json.open("w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    colours = {"all": "#2166ac", "body": "#4dac26", "last_only": "#d01c8b"}
    labels  = {"all": "All positions", "body": "Body\n(non-final)", "last_only": "Last\nonly"}

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharey=True)
    axes = axes.flatten()
    for ax, L in zip(axes, PATCH_LAYERS):
        for j, subset in enumerate(POSITION_SUBSETS):
            r = summary.get(L, {}).get(subset)
            if r is None:
                continue
            ax.bar(
                j, r["mean_delta"],
                color=colours[subset], alpha=0.85, width=0.65,
                yerr=r["sem_delta"],
                error_kw={"elinewidth": 1.2, "capsize": 3},
            )
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.set_title(f"Patch at L{L}", fontsize=8)
        ax.set_xticks(range(len(POSITION_SUBSETS)))
        ax.set_xticklabels([labels[s] for s in POSITION_SUBSETS],
                           rotation=20, ha="right", fontsize=7)

    axes[0].set_ylabel(
        f"Mean Δ projection (L{MEASUREMENT_LAYER} mean-token)\n"
        "negative = shifted toward secure"
    )
    fig.suptitle(
        "Activation patching: causal test of positional distribution\n"
        "(mean ± SEM across pairs; negative = representation shifted toward secure)",
        fontsize=8,
    )
    fig.tight_layout()
    fig_path = PAPER_FIGS / "fig_causal_patching.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {fig_path}")


if __name__ == "__main__":
    main()

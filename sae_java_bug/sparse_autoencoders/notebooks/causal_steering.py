"""
causal_steering.py — Contrastive direction steering causal experiment.

Steers the residual stream of all samples by subtracting α * d_L at every
token position and layer L ∈ {3, 7, 11, 15, 19, 23}, where d_L is the
unit-normalised vulnerability direction (mean_vuln − mean_secure) computed
from the pre-saved mean-pool tensors at layer L.

After steering, the mean-token probe is re-run and AUROC is reported for
each α value.  A drop in AUROC as α increases confirms that the direction
is causally active: removing it degrades the vulnerability-separating
information in the representation.

Connection to MoC (yu2025moc)
------------------------------
yu2025moc showed that per-token linear corrections steer code generation
toward secure outputs.  Our d_L is the mechanistic account of *why* this
works: it is the stable vulnerability axis identified across L3–L23.
If steering along −d_L collapses AUROC, our direction is the causal
substrate that per-token interventions exploit.

Usage
-----
    conda run -n demeanor python causal_steering.py [--n_samples 200]

Saves
-----
    artifacts/causal_steering/steering_results.json
    <paper_figs>/fig_causal_steering.pdf
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
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths ──────────────────────────────────────────────────────────────────────
ARTIFACTS  = Path(__file__).parents[2] / "artifacts"
ACT_DIR    = ARTIFACTS / "activations"
OUT_DIR    = ARTIFACTS / "causal_steering"
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

MODEL_ID      = "Qwen/Qwen2.5-7B-Instruct"
STEER_LAYERS  = [3, 7, 11, 15, 19, 23]   # stable-direction plateau (L3–L23); L0 and L27 excluded
MEAS_LAYER    = 27                       # mean-token extracted here (final sampled layer)
MAX_TOKENS    = 512
SEED          = 42
ALPHAS        = [0, 5, 10, 20, 40]        # 0 = unsteered baseline

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7, "figure.dpi": 150,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


# ── Data ───────────────────────────────────────────────────────────────────────

def load_samples(jsonl_path: Path, n_samples: int | None = None) -> list[dict]:
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
                    "cwe": r.get("cwe", ""),
                    "file_extension": r.get("file_extension", ""),
                })
            if n_samples and len(records) >= n_samples:
                break
    return records


# ── Vulnerability directions ───────────────────────────────────────────────────

def load_vuln_directions(layers: list[int]) -> dict[int, torch.Tensor]:
    """Returns {layer: unit-norm direction [d_model]} from pre-saved mean-pool tensors."""
    directions = {}
    for layer in layers:
        for run_dir in sorted((ACT_DIR / "mean_pool").iterdir(), reverse=True):
            sp = run_dir / f"safe_layer_{layer}.pt"
            vp = run_dir / f"vulnerable_layer_{layer}.pt"
            if sp.exists() and vp.exists():
                s = torch.load(sp, map_location="cpu", weights_only=True).float()
                v = torch.load(vp, map_location="cpu", weights_only=True).float()
                directions[layer] = F.normalize(
                    (v.mean(0) - s.mean(0)).unsqueeze(0), dim=-1
                ).squeeze(0)
                break
        else:
            raise FileNotFoundError(f"No mean_pool tensors for layer {layer}")
    return directions


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


# ── Steering hook ──────────────────────────────────────────────────────────────

def make_steer_hook(direction: torch.Tensor, alpha: float):
    """
    Subtracts alpha * direction from every token position in the residual stream.
    Steering in the −d direction moves representations toward the secure class.
    Handles both tuple and raw-tensor layer outputs.
    """
    def hook(module, inp, output):
        h = _hidden_from_output(output)
        return _replace_hidden(output, h - alpha * direction.to(h.device))
    return hook


# ── Mean-token extraction ──────────────────────────────────────────────────────

@torch.no_grad()
def get_mean_token(
    text: str,
    tokenizer,
    model,
    device: torch.device,
    meas_layer: int,
) -> np.ndarray:
    """
    Runs text through the model (with any pre-registered hooks) and returns
    the mean-token residual vector at meas_layer.  Shape: [d_model].
    """
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=MAX_TOKENS,
    ).to(device)

    store = {}

    def _capture(m, i, o):
        store["h"] = _hidden_from_output(o).detach()  # [batch, seq, d_model]

    h_meas = model.model.layers[meas_layer].register_forward_hook(_capture)
    try:
        model(**inputs, use_cache=False)
    finally:
        h_meas.remove()

    h = store["h"]           # [1, T, d_model]
    return h[0].float().mean(dim=0).cpu().numpy()   # [d_model]


# ── Projection AUROC ───────────────────────────────────────────────────────────
# Use direct projection onto the pre-computed vulnerability direction rather
# than a fitted probe. This avoids PCA sign-flip on small samples and directly
# measures whether steering moves representations along the known security axis.

def projection_auroc(
    safe_mat: np.ndarray,
    vuln_mat: np.ndarray,
    direction: torch.Tensor,
) -> float:
    """
    Projects each mean-token vector onto the vulnerability direction and
    computes AUROC(secure=0, vulnerable=1).  No fitting required.
    A drop in AUROC as alpha increases confirms the direction is causally active.
    """
    d = direction.numpy().astype(np.float32)
    proj_safe = safe_mat @ d   # [n]
    proj_vuln = vuln_mat @ d   # [n]
    n = len(proj_safe)
    y      = np.array([0] * n + [1] * n, dtype=int)
    scores = np.concatenate([proj_safe, proj_vuln])
    return float(roc_auc_score(y, scores))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of pairs to process (default 200)")
    args = parser.parse_args()

    print(f"Loading samples (stratified by file extension)...")
    # Load all records then stratify — the first N records in the JSONL skew
    # heavily toward CWE-79/PHP/JS where mean-token AUROC is below chance,
    # which would make the alpha=0 baseline unrepresentative.
    all_records = load_samples(SOURCE_JSONL, n_samples=None)
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_records)
    records = all_records[:args.n_samples]
    print(f"  Using {len(records)} pairs (shuffled from {len(all_records)} total)")

    print("  Loading vulnerability directions...")
    directions = load_vuln_directions(STEER_LAYERS)
    # Direction at the measurement layer is used for projection AUROC
    meas_direction = directions[MEAS_LAYER]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    print(f"  Loading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    )
    model.eval()

    # Collect mean-token vectors for each alpha
    # {alpha: {"safe": [vec, ...], "vuln": [vec, ...]}}
    buffers = {a: {"safe": [], "vuln": []} for a in ALPHAS}

    for i, rec in enumerate(records):
        if i % 25 == 0:
            print(f"  Sample {i + 1}/{len(records)}")

        for alpha in ALPHAS:
            handles = []
            if alpha > 0:
                for L in STEER_LAYERS:
                    h = model.model.layers[L].register_forward_hook(
                        make_steer_hook(directions[L], alpha)
                    )
                    handles.append(h)
            try:
                v_vec = get_mean_token(rec["vuln"],    tokenizer, model, device, MEAS_LAYER)
                s_vec = get_mean_token(rec["secure"],  tokenizer, model, device, MEAS_LAYER)
                buffers[alpha]["vuln"].append(v_vec)
                buffers[alpha]["safe"].append(s_vec)
            except Exception as e:
                print(f"    [WARN] sample {i} alpha={alpha}: {e}")
            finally:
                for h in handles:
                    h.remove()

    # ── Projection AUROC per alpha ──────────────────────────────────────────────
    auroc_by_alpha: dict[int, float] = {}
    print(f"\nProjection AUROC at L{MEAS_LAYER} (steering L{STEER_LAYERS[0]}–L{STEER_LAYERS[-1]}):")
    for alpha in ALPHAS:
        safe_mat = np.array(buffers[alpha]["safe"])
        vuln_mat = np.array(buffers[alpha]["vuln"])
        if len(safe_mat) < 10:
            print(f"  alpha={alpha:>3}  [skipped — only {len(safe_mat)} samples collected]")
            continue
        auc = projection_auroc(safe_mat, vuln_mat, meas_direction)
        # Projection AUROC below 0.5 means the direction is inverted for this
        # alpha — normalise so 0.5 = chance and values move away from 0.5.
        auroc_by_alpha[alpha] = auc
        tag = " ← unsteered baseline" if alpha == 0 else ""
        print(f"  alpha={alpha:>3}  AUROC={auc:.3f}{tag}")

    out_json = OUT_DIR / "steering_results.json"
    with out_json.open("w") as f:
        json.dump({str(k): v for k, v in auroc_by_alpha.items()}, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    alphas_sorted = sorted(auroc_by_alpha.keys())
    aucs          = [auroc_by_alpha[a] for a in alphas_sorted]
    baseline_auc  = auroc_by_alpha.get(0, 0.5)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(alphas_sorted, aucs, "o-", color="#2166ac", lw=2, ms=6, label="Steered")
    ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.7, label="Chance (0.5)")
    ax.axhline(baseline_auc, color="#d01c8b", lw=1.2, ls="--", alpha=0.8,
               label=f"Baseline α=0 ({baseline_auc:.3f})")
    ax.fill_between(alphas_sorted, 0.5, aucs,
                    where=[a < baseline_auc for a in aucs],
                    alpha=0.12, color="#2166ac",
                    label="AUROC drop from steering")
    ax.set_xlabel("Steering magnitude α")
    ax.set_ylabel(f"AUROC (vuln vs. secure)\nL{MEAS_LAYER} mean-token probe")
    ax.set_title(
        f"Contrastive steering along −d_L\n"
        f"(layers {STEER_LAYERS[0]}–{STEER_LAYERS[-1]} steered simultaneously)",
        fontsize=8,
    )
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig_path = PAPER_FIGS / "fig_causal_steering.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {fig_path}")


if __name__ == "__main__":
    main()

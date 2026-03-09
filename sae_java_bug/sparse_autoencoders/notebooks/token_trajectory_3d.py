"""
token_trajectory_3d.py

Per-token residual-stream trajectory visualisation in guided 3-D PCA space.

For a given layer L, constructs a coordinate system where:
  - x-axis = vulnerability direction d^L (forced)
              (precomputed from mean_pool activations; within-C version)
  - y,z-axes = top 2 PCA components orthogonal to d^L

For each selected (vulnerable, secure) code pair, plots the sequence of
per-token hidden states as a trajectory through this space, coloured by
token position (light → dark).

Expected observation under negative-space hypothesis:
  - Secure code trajectories accumulate activation in the -d^L direction
    (defensive constructs push the representation away from "vulnerable")
  - Vulnerable code trajectories stay closer to d^L or drift less

Per-token activations are expensive to obtain (need model inference), so
results are cached to:
  artifacts/activations/token_traj/layer{L}_n{N}.npz

Usage:
    conda run -n sae python token_trajectory_3d.py              # defaults
    conda run -n sae python token_trajectory_3d.py --layer 15
    conda run -n sae python token_trajectory_3d.py --no_cache   # force re-run
"""

import argparse
import ctypes
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from sklearn.decomposition import PCA as skPCA

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
ARTIFACTS  = Path(__file__).parents[2] / "artifacts" / "activations"
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)
CACHE_DIR  = ARTIFACTS / "token_traj"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LAYERS     = [0, 3, 7, 11, 15, 19, 23, 27]
SEED       = 42
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
C_EXTS     = {"c", "cc", "cpp", "h"}

mpl.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "figure.dpi":      150,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})


# ── Tensor helpers ────────────────────────────────────────────────────────────
def _t2np(t: "torch.Tensor") -> np.ndarray:
    """Fast tensor → numpy via ctypes (avoids Python-level bytes() copy)."""
    t = t.float().contiguous()
    buf = (ctypes.c_float * t.numel()).from_address(t.data_ptr())
    return np.ctypeslib.as_array(buf).reshape(t.shape).copy()


# ── Vulnerability direction ───────────────────────────────────────────────────
def load_vuln_direction(layer: int) -> np.ndarray:
    """
    Load precomputed mean_pool activations and return the within-C
    vulnerability direction d^L = normalize(mean_vuln - mean_safe).
    """
    runs = sorted((ARTIFACTS / "mean_pool").glob("*/meta.json"))
    if not runs:
        raise FileNotFoundError(
            "No mean_pool run found. Run mean_pool_probe.py first."
        )
    run_dir = runs[-1].parent

    with (run_dir / "meta.json").open() as f:
        meta = json.load(f)

    safe = _t2np(torch.load(run_dir / f"safe_layer_{layer}.pt",       weights_only=True))
    vuln = _t2np(torch.load(run_dir / f"vulnerable_layer_{layer}.pt", weights_only=True))

    mask = np.array([r["file_extension"] in C_EXTS for r in meta])
    safe, vuln = safe[mask], vuln[mask]

    d = vuln.mean(0) - safe.mean(0)
    return d / np.linalg.norm(d)


# ── Sample selection ──────────────────────────────────────────────────────────
JSONL_SOURCE = (
    ARTIFACTS / "raw_activations"
    / "vulnerable_code_qwen_coder_standard_16384_raw"
    / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
)


def select_samples(
    n: int = 5,
    max_chars: int = 1500,
    seed: int = SEED,
) -> tuple[list[str], list[str], list[str]]:
    """
    Load records from local JSONL and select n diverse C-language pairs.
    Fields: vulnerable_code / secure_code (base64), file_extension, cwe.
    Filters by decoded code length to keep inference fast.
    Returns (vuln_texts, safe_texts, cwe_labels).
    """
    import base64

    print(f"Loading records from {JSONL_SOURCE.name} ...")
    records = []
    with JSONL_SOURCE.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("file_extension", "") not in C_EXTS:
                continue
            try:
                vc = base64.b64decode(r["vulnerable_code"]).decode("utf-8", errors="replace")
                sc = base64.b64decode(r["secure_code"]).decode("utf-8", errors="replace")
            except Exception:
                vc = r.get("vulnerable_code", "")
                sc = r.get("secure_code", "")
            if not (100 < len(vc) < max_chars and 100 < len(sc) < max_chars):
                continue
            records.append({"vc": vc, "sc": sc, "cwe": str(r.get("cwe", "?"))})

    print(f"  {len(records)} eligible C pairs (length 100–{max_chars} chars).")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(records), size=min(n, len(records)), replace=False)
    vuln_texts = [records[i]["vc"]  for i in chosen]
    safe_texts = [records[i]["sc"]  for i in chosen]
    cwes       = [records[i]["cwe"] for i in chosen]

    for i, (cwe, vt, st) in enumerate(zip(cwes, vuln_texts, safe_texts)):
        print(f"  Pair {i+1}: {cwe}  vuln={len(vt)} chars, safe={len(st)} chars")

    return vuln_texts, safe_texts, cwes


# ── Model inference ───────────────────────────────────────────────────────────
def extract_per_token(
    texts:      list[str],
    layer:      int,
    max_length: int = 350,
) -> list[np.ndarray]:
    """
    Run Qwen2.5-7B-Instruct on each text and return per-token hidden states
    at the given layer as a list of [T_i, 3584] numpy arrays.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for i, text in enumerate(texts):
        print(f"  [{i+1}/{len(texts)}] tokenising and running forward pass ...")
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        T = enc["input_ids"].shape[1]
        print(f"        T = {T} tokens")

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

        # hidden_states: tuple of (n_layers+1) tensors, each [1, T, D]
        # hidden_states[0]   = embedding
        # hidden_states[L+1] = after transformer block L
        h = out.hidden_states[layer + 1]  # [1, T, D]
        results.append(_t2np(h[0]))       # [T, D]

    print("Inference complete.")
    del model
    return results


# ── Cache ─────────────────────────────────────────────────────────────────────
def cache_file(layer: int, n: int) -> Path:
    return CACHE_DIR / f"layer{layer}_n{n}.npz"


def save_cache(
    vuln_acts: list[np.ndarray],
    safe_acts: list[np.ndarray],
    cwes: list[str],
    layer: int,
):
    path = cache_file(layer, len(vuln_acts))
    kw = {"n": np.array(len(vuln_acts)), "cwes": np.array(cwes)}
    kw |= {f"v{i}": a for i, a in enumerate(vuln_acts)}
    kw |= {f"s{i}": a for i, a in enumerate(safe_acts)}
    np.savez(path, **kw)
    print(f"Cached → {path}")


def load_cache(
    layer: int,
    n: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]] | None:
    path = cache_file(layer, n)
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    m    = int(data["n"])
    cwes = list(data["cwes"])
    return [data[f"v{i}"] for i in range(m)], [data[f"s{i}"] for i in range(m)], cwes


# ── Guided PCA projection ─────────────────────────────────────────────────────
class GuidedPCA3D:
    """
    Projects activations into 3-D guided PCA space.
      axis 1 (x) = forced vulnerability direction d
      axes 2–3   = top 2 PCA components of the d-orthogonal residual,
                   fitted jointly on all supplied activations.
    """

    def __init__(self, direction: np.ndarray):
        self.d   = direction / np.linalg.norm(direction)
        self.mu  = None
        self.pc2 = None
        self.pc3 = None

    def fit(self, acts: np.ndarray):
        """Fit on all token activations jointly. acts: [N_total, D]"""
        self.mu  = acts.mean(0)
        X        = acts - self.mu
        # Project out d
        coeff    = X @ self.d                           # [N]
        X_resid  = X - np.outer(coeff, self.d)          # [N, D]
        # PCA on residual
        pca = skPCA(n_components=2, random_state=SEED)
        pca.fit(X_resid)
        self.pc2 = pca.components_[0]                   # [D]
        self.pc3 = pca.components_[1]                   # [D]
        var_d    = np.var(coeff)
        var_tot  = X.var(0).sum()
        print(f"  GuidedPCA: d-axis explains {100*var_d/var_tot:.1f}% of total variance")
        print(f"  PCA on residual: PC2 {100*pca.explained_variance_ratio_[0]:.1f}%,"
              f" PC3 {100*pca.explained_variance_ratio_[1]:.1f}%")
        return self

    def transform(self, act: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project [T, D] → (x, y, z) each [T]."""
        A = act - self.mu
        x = A @ self.d
        R = A - np.outer(x, self.d)
        y = R @ self.pc2
        z = R @ self.pc3
        return x, y, z


# ── Smooth helper ─────────────────────────────────────────────────────────────
def smooth(arr: np.ndarray, w: int = 5) -> np.ndarray:
    """Simple moving average for trajectory smoothing."""
    if len(arr) <= w:
        return arr
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="same")


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(
    vuln_acts:  list[np.ndarray],
    safe_acts:  list[np.ndarray],
    cwes:       list[str],
    direction:  np.ndarray,
    layer:      int,
    out_path:   Path,
    smooth_w:   int = 7,
    elev:       float = 20,
    azim:       float = -60,
):
    n = len(vuln_acts)

    # ── Fit guided PCA on all tokens jointly
    all_acts = np.vstack(vuln_acts + safe_acts)
    gpca = GuidedPCA3D(direction).fit(all_acts)

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig   = plt.figure(figsize=(5.5 * ncols, 4.5 * nrows))

    for i in range(n):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        ax.view_init(elev=elev, azim=azim)

        vx, vy, vz = gpca.transform(vuln_acts[i])
        sx, sy, sz = gpca.transform(safe_acts[i])

        # Smooth for cleaner trajectories
        vx, vy, vz = smooth(vx, smooth_w), smooth(vy, smooth_w), smooth(vz, smooth_w)
        sx, sy, sz = smooth(sx, smooth_w), smooth(sy, smooth_w), smooth(sz, smooth_w)

        T_v, T_s = len(vx), len(sx)

        # ── Draw trajectories with position-based colour
        for j in range(T_v - 1):
            t = j / T_v
            ax.plot(
                [vx[j], vx[j+1]], [vy[j], vy[j+1]], [vz[j], vz[j+1]],
                color=cm.Reds(0.35 + 0.55 * t),
                lw=0.9, alpha=0.75, solid_capstyle="round",
            )

        for j in range(T_s - 1):
            t = j / T_s
            ax.plot(
                [sx[j], sx[j+1]], [sy[j], sy[j+1]], [sz[j], sz[j+1]],
                color=cm.Blues(0.35 + 0.55 * t),
                lw=0.9, alpha=0.75, solid_capstyle="round",
            )

        # Start (●) and end (■) markers
        for xs, ys, zs, col, ecol in [
            ([vx[0]], [vy[0]], [vz[0]], cm.Reds(0.5),  "darkred"),
            ([vx[-1]], [vy[-1]], [vz[-1]], cm.Reds(0.9), "darkred"),
            ([sx[0]], [sy[0]], [sz[0]], cm.Blues(0.5),  "darkblue"),
            ([sx[-1]], [sy[-1]], [sz[-1]], cm.Blues(0.9), "darkblue"),
        ]:
            ax.scatter(xs, ys, zs, c=[col], s=25, zorder=5, edgecolors=ecol, linewidths=0.5)

        # ── Vulnerability direction arrow
        #    anchor at the centroid of the data in this panel
        cx = float(np.concatenate([vx, sx]).mean())
        cy = float(np.concatenate([vy, sy]).mean())
        cz = float(np.concatenate([vz, sz]).mean())
        sc = float(max(vx.std(), sx.std(), 0.01)) * 1.5
        ax.quiver(
            cx - sc * 0.5, cy, cz,
            sc, 0, 0,
            color="#888888", arrow_length_ratio=0.25,
            lw=1.4, alpha=0.65,
        )
        ax.text(cx + sc * 0.6, cy, cz, r"$\mathbf{d}^L$",
                color="#555555", fontsize=7, ha="left")

        ax.set_xlabel(r"$\leftarrow$ secure  |  $\mathbf{d}^L$  |  vuln $\rightarrow$",
                      fontsize=6.5, labelpad=-2)
        ax.set_ylabel("PC2\n(orthog.)", fontsize=6.5, labelpad=-2)
        ax.set_zlabel("PC3\n(orthog.)", fontsize=6.5, labelpad=-2)
        ax.tick_params(labelsize=5.5)
        ax.set_title(f"Pair {i+1} · {cwes[i]}", fontsize=8)

    # ── Shared legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elems = [
        Line2D([0], [0], color=cm.Reds(0.6),  lw=2, label="Vulnerable tokens"),
        Line2D([0], [0], color=cm.Blues(0.6), lw=2, label="Secure tokens"),
        Line2D([0], [0], color="none", label="light = sequence start"),
        Line2D([0], [0], color="none", label="dark  = sequence end"),
    ]
    fig.legend(handles=legend_elems, loc="lower center",
               ncol=4, fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        f"Per-token trajectories in guided 3-D PCA — Layer {layer}\n"
        r"$x$-axis forced = vulnerability direction $\mathbf{d}^L$  "
        r"(within-C mean$_{\rm vuln}$ − mean$_{\rm safe}$)",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer",      type=int, default=11,
                        help="Residual stream layer to visualise (default: 11)")
    parser.add_argument("--n_samples",  type=int, default=6,
                        help="Number of (vuln, secure) pairs to show")
    parser.add_argument("--max_chars",  type=int, default=1500,
                        help="Max code length in characters (controls inference speed)")
    parser.add_argument("--max_tokens", type=int, default=300,
                        help="Max tokens per forward pass")
    parser.add_argument("--smooth",     type=int, default=7,
                        help="Trajectory smoothing window (tokens)")
    parser.add_argument("--no_cache",   action="store_true",
                        help="Force re-run inference even if cache exists")
    args = parser.parse_args()

    layer = args.layer
    print(f"\n=== Token trajectory visualisation — Layer L{layer} ===\n")

    # 1. Vulnerability direction from precomputed mean_pool data
    print("Loading vulnerability direction d^L (within-C) ...")
    direction = load_vuln_direction(layer)
    print(f"  d^L computed (norm = {np.linalg.norm(direction):.4f})")

    # 2. Per-token activations (cached or fresh)
    cached = None if args.no_cache else load_cache(layer, args.n_samples)

    if cached is not None:
        vuln_acts, safe_acts, cwes = cached
        print(f"Using cached activations ({len(vuln_acts)} pairs).")
    else:
        vuln_texts, safe_texts, cwes = select_samples(
            n=args.n_samples, max_chars=args.max_chars
        )
        print(f"\nRunning model inference (L{layer}) ...")
        all_acts  = extract_per_token(
            vuln_texts + safe_texts, layer=layer, max_length=args.max_tokens
        )
        vuln_acts = all_acts[:args.n_samples]
        safe_acts = all_acts[args.n_samples:]
        save_cache(vuln_acts, safe_acts, cwes, layer)

    # 3. Visualise
    out_path = PAPER_FIGS / f"fig_token_trajectory_3d_L{layer}.pdf"
    print(f"\nBuilding 3-D figure → {out_path}")
    make_figure(
        vuln_acts, safe_acts, cwes,
        direction, layer, out_path,
        smooth_w=args.smooth,
    )


if __name__ == "__main__":
    main()

"""
causal_patching_behavioral.py — Downstream behavioral consequences of activation patching.

Extends causal_patching.py with two logit-level measurements that answer the
reviewer concern: "patching demonstrates representation movement, but downstream
behavioral consequences are not shown."

Experiment A — Log-probability of secure tokens
    For each (vulnerable, secure) pair, body-patch the vulnerable forward pass at
    each of the 8 sampled layers, then measure the mean per-token log-probability
    of the SECURE code tokens.  A positive delta means the patched model assigns
    higher likelihood to the secure token sequence than the unpatched baseline.

Experiment C — Security vocabulary logit shift
    At the final token position of each forward pass, measure the mean log-softmax
    score over a predefined security vocabulary (bounds-check, null-check, input-
    validation, error-handling tokens).  A positive delta means body patching makes
    the model more likely to predict a security-relevant token as the next token.

Both experiments share the same forward passes:
    per pair: 1 secure-cache pass + 1 baseline pass + 8 body-patched passes = 10 total
    (vs. the original causal_patching.py: 2 + 8×3 = 26 passes per pair)

Usage
-----
    conda run -n sae python causal_patching_behavioral.py [--n_pairs 100]

Saves
-----
    artifacts/causal_patching/behavioral_results.json
    <paper_figs>/fig_behavioral_patching.pdf
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
ARTIFACTS = Path(__file__).parents[2] / "artifacts"
ACT_DIR = ARTIFACTS / "activations"
OUT_DIR = ARTIFACTS / "causal_patching"
SOURCE_JSONL = (
    ACT_DIR
    / "raw_activations"
    / "vulnerable_code_qwen_coder_standard_16384_raw"
    / "activations_layer_0_raw_component_hidden_state_last_token.jsonl"
)
PAPER_FIGS = (
    Path(__file__).parents[4]
    / "On-the-Absence-of-Global-Anomalies-in-Vulnerable-Code-Representations"
    / "figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PATCH_LAYERS = [0, 3, 7, 11, 15, 19, 23, 27]
MAX_TOKENS = 512
SEED = 42

# Security-relevant token strings for Experiment C.
# Only strings that tokenize as a single token (model-dependent) will be used.
SECURITY_VOCAB_STRINGS = [
    # C / C++ memory safety
    "NULL",
    "nullptr",
    "assert",
    "bounds",
    "sizeof",
    "malloc",
    "calloc",
    "realloc",
    "free",
    # Input validation (language-agnostic)
    "validate",
    "valid",
    "check",
    "verify",
    "sanitize",
    "sanitise",
    "filter",
    "escape",
    "encode",
    # Error handling
    "error",
    "Error",
    "errno",
    "throw",
    "catch",
    # PHP / web security
    "htmlspecialchars",
    "htmlentities",
    "PDO",
    "bindParam",
    "bindValue",
    "prepared",
    "prepare",
    # JavaScript security
    "parseInt",
    "isNaN",
    "isFinite",
    "encodeURIComponent",
]

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.dpi": 150,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


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
                sec = base64.b64decode(r["secure_code"]).decode(
                    "utf-8", errors="replace"
                )
                vuln = base64.b64decode(r["vulnerable_code"]).decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                sec = r.get("secure_code", "")
                vuln = r.get("vulnerable_code", "")
            if sec.strip() and vuln.strip():
                records.append(
                    {
                        "secure": sec,
                        "vuln": vuln,
                        "cwe": r.get("cwe", ""),
                        "ext": r.get("ext", ""),
                    }
                )
            if n_pairs and len(records) >= n_pairs:
                break
    return records


def get_security_vocab_ids(tokenizer) -> list[int]:
    """
    Return unique token IDs for SECURITY_VOCAB_STRINGS that encode as a single
    token.  Tries both the bare string and a space-prefixed version (for BPE
    tokenisers that represent mid-word occurrences without a leading space).
    """
    ids: set[int] = set()
    for s in SECURITY_VOCAB_STRINGS:
        for candidate in [s, " " + s]:
            toks = tokenizer.encode(candidate, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
    return sorted(ids)


# ── Hook utilities (mirrored from causal_patching.py) ─────────────────────────


def _hidden_from_output(output):
    return output if isinstance(output, torch.Tensor) else output[0]


def _replace_hidden(output, new_hidden):
    if isinstance(output, torch.Tensor):
        return new_hidden
    out = list(output)
    out[0] = new_hidden
    return tuple(out)


def make_cache_hook(store: dict, key):
    def hook(module, inp, output):
        store[key] = _hidden_from_output(output).detach().clone()

    return hook


def make_body_patch_hook(
    secure_act: torch.Tensor, layer_id: int = None, debug: bool = False
):
    """Patch all shared-prefix positions except the last token (body subset)."""

    def hook(module, inp, output):
        h = _hidden_from_output(output).clone()
        T_v = h.shape[1]
        T_s = secure_act.shape[1]
        src = secure_act.to(device=h.device, dtype=h.dtype)
        end = max(min(T_v, T_s) - 1, 0)  # body = shared prefix minus final token
        if debug and layer_id is not None:
            # Check if patch is actually changing anything
            before = h[0, :5, :5].clone()
            h[0, :end, :] = src[0, :end, :]
            after = h[0, :5, :5].clone()
            print(f"[DEBUG L{layer_id}] Patch applied: T_v={T_v}, T_s={T_s}, end={end}")
            print(f"  Before patch [0, 0, :5]: {before[0, :5]}")
            print(f"  After patch [0, 0, :5]: {after[0, :5]}")
            print(f"  Max diff: {(after - before).abs().max().item():.6f}")
        else:
            if end > 0:
                h[0, :end, :] = src[0, :end, :]
        return _replace_hidden(output, h)

    return hook


# ── Core measurement ───────────────────────────────────────────────────────────


@torch.no_grad()
def cache_secure_layers(
    sec_ids: torch.Tensor,
    model,
    device: torch.device,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """One forward pass through the secure sample; caches residuals at all patch layers."""
    inp = sec_ids[:, :MAX_TOKENS].to(device)
    store: dict[int, torch.Tensor] = {}
    handles = [
        model.model.layers[L].register_forward_hook(make_cache_hook(store, L))
        for L in layers
    ]
    try:
        model(input_ids=inp, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return store


@torch.no_grad()
def measure_forward(
    vuln_ids: torch.Tensor,  # [1, T_v]
    sec_ids: torch.Tensor,  # [1, T_s]  — used as labels for Exp A
    model,
    device: torch.device,
    vocab_ids: list[int],
    patch_layer: int | None = None,
    secure_cache: dict | None = None,
) -> tuple[float, float]:
    """
    Single forward pass returning:
      exp_a : mean per-token log-prob of sec_ids tokens given vuln_ids context
              (higher = model more likely to generate the secure sequence)
      exp_c : mean log-softmax score over security vocabulary at the final position
              (higher = model more likely to predict a security token next)

    If patch_layer and secure_cache are provided, body-patches the forward pass.
    """
    T = min(vuln_ids.shape[1], sec_ids.shape[1], MAX_TOKENS)
    inp = vuln_ids[:, :T].to(device)
    # labels: HuggingFace causal-LM shifts internally, so labels[i] = target at pos i
    labels = sec_ids[:, :T].to(device)

    handles = []
    if patch_layer is not None and secure_cache is not None:
        handles.append(
            model.model.layers[patch_layer].register_forward_hook(
                make_body_patch_hook(
                    secure_cache[patch_layer], layer_id=patch_layer, debug=False
                )
            )
        )

    try:
        out = model(input_ids=inp, labels=labels, use_cache=False)
        # Exp A: model.loss = mean cross-entropy = -mean log-prob
        exp_a = float(-out.loss.cpu())

        # Exp C: log-softmax at the final token position
        log_probs = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
        v_ids = torch.tensor(vocab_ids, device=log_probs.device)
        exp_c = float(log_probs[v_ids].mean().cpu()) if len(v_ids) > 0 else float("nan")
    finally:
        for h in handles:
            h.remove()

    return exp_a, exp_c


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral consequences of activation patching (Exp A + C)."
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=100,
        help="Number of pairs to evaluate (default 100)",
    )
    args = parser.parse_args()

    print("Loading pairs...")
    all_pairs = load_pairs(SOURCE_JSONL)
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_pairs)
    pairs = all_pairs[: args.n_pairs]
    print(f"  Using {len(pairs)} of {len(all_pairs)} pairs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    print(f"  Loading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    vocab_ids = get_security_vocab_ids(tokenizer)
    print(f"  Security vocabulary: {len(vocab_ids)} single-token entries")
    if vocab_ids:
        examples = [tokenizer.decode([i]).strip() for i in vocab_ids[:12]]
        print(f"  Examples: {examples}")

    # Storage: per layer, lists of per-pair deltas
    deltas: dict[int, dict[str, list[float]]] = {
        L: {"exp_a": [], "exp_c": []} for L in PATCH_LAYERS
    }
    # Also store baselines for sanity-check reporting
    base_a_all: list[float] = []
    base_c_all: list[float] = []

    for i, pair in enumerate(pairs):
        if i % 10 == 0:
            print(f"  Pair {i + 1}/{len(pairs)}", flush=True)
        try:
            vuln_ids = tokenizer(
                pair["vuln"],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKENS,
            ).input_ids
            sec_ids = tokenizer(
                pair["secure"],
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKENS,
            ).input_ids

            # Pass 1: cache secure activations at all patch layers
            sec_cache = cache_secure_layers(sec_ids, model, device, PATCH_LAYERS)

            # Pass 2: baseline (no patch)
            base_a, base_c = measure_forward(
                vuln_ids, sec_ids, model, device, vocab_ids
            )
            base_a_all.append(base_a)
            base_c_all.append(base_c)

            # Passes 3–10: one body-patched pass per layer
            for L in PATCH_LAYERS:
                patch_a, patch_c = measure_forward(
                    vuln_ids,
                    sec_ids,
                    model,
                    device,
                    vocab_ids,
                    patch_layer=L,
                    secure_cache=sec_cache,
                )
                deltas[L]["exp_a"].append(patch_a - base_a)
                deltas[L]["exp_c"].append(patch_c - base_c)

        except Exception as e:
            print(f"    [WARN] pair {i}: {e}")

    # ── Summarise ──────────────────────────────────────────────────────────────
    summary: dict = {
        "baseline": {
            "exp_a_mean": float(np.mean(base_a_all)) if base_a_all else None,
            "exp_c_mean": float(np.mean(base_c_all)) if base_c_all else None,
            "n": len(base_a_all),
        },
        "vocab_size": len(vocab_ids),
        "vocab_examples": [tokenizer.decode([i]).strip() for i in vocab_ids[:20]],
    }

    print(f"\nBaseline (unpatched vulnerable):")
    print(f"  Exp A mean log-prob: {summary['baseline']['exp_a_mean']:.4f}")
    print(f"  Exp C vocab score:   {summary['baseline']['exp_c_mean']:.4f}")
    print(
        f"\n{'Layer':<7} {'ΔExp A (log-prob secure)':<28} {'ΔExp C (vocab score)':<26}"
    )
    print("-" * 62)

    for L in PATCH_LAYERS:
        a = np.array(deltas[L]["exp_a"])
        c = np.array(deltas[L]["exp_c"])
        n = len(a)
        if n == 0:
            continue
        a_mean, a_sem = float(a.mean()), float(a.std() / np.sqrt(n))
        c_mean, c_sem = float(c.mean()), float(c.std() / np.sqrt(n))
        summary[L] = {
            "exp_a_mean": a_mean,
            "exp_a_sem": a_sem,
            "exp_c_mean": c_mean,
            "exp_c_sem": c_sem,
            "n": n,
        }
        sig_a = "*" if abs(a_mean) > 2 * a_sem else " "
        sig_c = "*" if abs(c_mean) > 2 * c_sem else " "
        print(
            f"  L{L:<4} {a_mean:+.4f} ± {a_sem:.4f} {sig_a}   {c_mean:+.4f} ± {c_sem:.4f} {sig_c}"
        )

    print("  (* = |mean| > 2 SEM)")

    out_json = OUT_DIR / "behavioral_results.json"
    with out_json.open("w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    xs = list(range(len(PATCH_LAYERS)))
    x_lbls = [f"L{L}" for L in PATCH_LAYERS]
    colour = "#4dac26"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))

    for ax, key, title, ylabel in [
        (
            ax1,
            "exp_a",
            "Exp A: Secure token log-prob shift",
            "Δ mean log-prob(secure tokens | patched vuln)\n"
            "positive → model more likely to generate secure sequence",
        ),
        (
            ax2,
            "exp_c",
            "Exp C: Security vocabulary logit shift",
            "Δ mean log-softmax(security vocab, final pos)\n"
            "positive → higher prob of security token as next token",
        ),
    ]:
        means = [summary.get(L, {}).get(f"{key}_mean", 0) or 0 for L in PATCH_LAYERS]
        sems = [summary.get(L, {}).get(f"{key}_sem", 0) or 0 for L in PATCH_LAYERS]
        bars = ax.bar(
            xs,
            means,
            color=colour,
            alpha=0.80,
            width=0.60,
            yerr=sems,
            error_kw={"elinewidth": 1.2, "capsize": 3},
        )
        # Colour bars below zero in a lighter shade to flag negative effects
        for bar, m in zip(bars, means):
            if m < 0:
                bar.set_facecolor("#d95f02")
                bar.set_alpha(0.75)
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels(x_lbls, rotation=45, ha="right")
        ax.set_xlabel("Patch layer")
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.set_title(title, fontweight="bold")

    n_reported = summary["baseline"]["n"]
    fig.suptitle(
        f"Behavioral consequences of body-patch activation patching  (n={n_reported} pairs)\n"
        "Baseline = unpatched vulnerable forward pass; patch = secure body activations injected",
        fontsize=8.5,
    )
    fig.tight_layout()

    fig_path = PAPER_FIGS / "fig_behavioral_patching.pdf"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {fig_path}")


if __name__ == "__main__":
    main()

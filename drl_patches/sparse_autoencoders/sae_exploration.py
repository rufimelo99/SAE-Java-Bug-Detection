import pandas as pd
import torch
from sae_lens import SAE, HookedSAETransformer
from tqdm import trange

from drl_patches.logger import logger
from drl_patches.sparse_autoencoders.analyse_layers import store_values
from drl_patches.sparse_autoencoders.schemas import PlotType

torch.set_grad_enabled(False)
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("Getting device.", device=device)

RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.0.hook_resid_pre"
HOOK_POINT = ...  # "residuals"
MODEL_NAME = "gpt2-small"  # "meta-llama/Llama-3.1-8B"
MSR_df = pd.read_csv("artifacts/gbug-java.csv")

model = HookedSAETransformer.from_pretrained(MODEL_NAME, device=device)

sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=RELEASE,
    sae_id=SAE_ID,
    device=device,
)

for i in trange(len(MSR_df)):
    prompt = [
        MSR_df.iloc[i]["func_before"],
        MSR_df.iloc[i]["func_after"],
    ]
    _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])

    index = [f"feature_{i}" for i in range(sae.cfg.d_sae)]
    feature_activation_df = pd.DataFrame(
        cache["blocks.0.hook_resid_pre.hook_sae_acts_post"][0, -1, :].cpu().numpy(),
        index=index,
    )
    feature_activation_df.columns = ["vulnerable"]
    feature_activation_df["secure"] = (
        cache["blocks.0.hook_resid_pre.hook_sae_acts_post"][1, -1, :].cpu().numpy()
    )
    feature_activation_df["diff"] = abs(
        feature_activation_df["vulnerable"] - feature_activation_df["secure"]
    )
    safe_values = feature_activation_df["secure"].values
    vuln_values = feature_activation_df["vulnerable"].values
    diff_values = feature_activation_df["diff"].values

    store_values(
        "artifacts/accumulated_featuree_importance_safe.jsonl",
        append=True,
        index=i,
        model=MODEL_NAME,
        logit_lens_logit_diffs=safe_values.tolist(),
        labels=index,
        plot_type=PlotType.SAE_FEATURE_IMPORTANCE,
    )

    store_values(
        "artifacts/accumulated_featuree_importance_vuln.jsonl",
        append=True,
        index=i,
        model=MODEL_NAME,
        logit_lens_logit_diffs=vuln_values.tolist(),
        labels=index,
        plot_type=PlotType.SAE_FEATURE_IMPORTANCE,
    )

    store_values(
        "artifacts/accunuated_featuree_importance_diff.jsonl",
        append=True,
        index=i,
        model=MODEL_NAME,
        logit_lens_logit_diffs=diff_values.tolist(),
        labels=index,
        plot_type=PlotType.SAE_FEATURE_IMPORTANCE,
    )
    breakpoint()

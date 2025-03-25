import argparse
import os
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import torch
from drl_patches.logger import logger
from drl_patches.sparse_autoencoders.analyse_layers import store_values
from drl_patches.sparse_autoencoders.schemas import AvailableModels, PlotType
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm, trange

torch.set_grad_enabled(False)
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
logger.info("Getting device.", device=device)


class Release(str, Enum):
    GPT2_SMALL_RES_JB = "gpt2-small-res-jb"
    GEMMA_SCOPE = "gemma-scope-2b-pt-res-canonical"
    LLAMA_SCOPE = "llama_scope_lxr_32x"


class SAE_ID(str, Enum):
    BLOCKS_0_HOOK_RESID_PRE = "blocks.0.hook_resid_pre"
    BLOCKS_1_HOOK_RESID_PRE = "blocks.1.hook_resid_pre"
    BLOCKS_2_HOOK_RESID_PRE = "blocks.2.hook_resid_pre"
    BLOCKS_3_HOOK_RESID_PRE = "blocks.3.hook_resid_pre"
    BLOCKS_4_HOOK_RESID_PRE = "blocks.4.hook_resid_pre"
    BLOCKS_5_HOOK_RESID_PRE = "blocks.5.hook_resid_pre"
    BLOCKS_6_HOOK_RESID_PRE = "blocks.6.hook_resid_pre"
    BLOCKS_7_HOOK_RESID_PRE = "blocks.7.hook_resid_pre"
    BLOCKS_8_HOOK_RESID_PRE = "blocks.8.hook_resid_pre"
    BLOCKS_9_HOOK_RESID_PRE = "blocks.9.hook_resid_pre"
    BLOCKS_10_HOOK_RESID_PRE = "blocks.10.hook_resid_pre"
    BLOCKS_11_HOOK_RESID_PRE = "blocks.11.hook_resid_pre"
    GEMMA_SCOPE_0_WIDTH_16K_CANONICAL = "layer_0/width_16k/canonical"
    GEMMA_SCOPE_1_WIDTH_16K_CANONICAL = "layer_1/width_16k/canonical"
    GEMMA_SCOPE_2_WIDTH_16K_CANONICAL = "layer_2/width_16k/canonical"
    GEMMA_SCOPE_3_WIDTH_16K_CANONICAL = "layer_3/width_16k/canonical"
    GEMMA_SCOPE_4_WIDTH_16K_CANONICAL = "layer_4/width_16k/canonical"
    GEMMA_SCOPE_5_WIDTH_16K_CANONICAL = "layer_5/width_16k/canonical"
    GEMMA_SCOPE_6_WIDTH_16K_CANONICAL = "layer_6/width_16k/canonical"
    GEMMA_SCOPE_7_WIDTH_16K_CANONICAL = "layer_7/width_16k/canonical"
    GEMMA_SCOPE_8_WIDTH_16K_CANONICAL = "layer_8/width_16k/canonical"
    GEMMA_SCOPE_9_WIDTH_16K_CANONICAL = "layer_9/width_16k/canonical"
    GEMMA_SCOPE_10_WIDTH_16K_CANONICAL = "layer_10/width_16k/canonical"
    GEMMA_SCOPE_11_WIDTH_16K_CANONICAL = "layer_11/width_16k/canonical"
    GENMA_SCOPE_12_WIDTH_16K_CANONICAL = "layer_12/width_16k/canonical"
    GENMA_SCOPE_13_WIDTH_16K_CANONICAL = "layer_13/width_16k/canonical"
    GENMA_SCOPE_14_WIDTH_16K_CANONICAL = "layer_14/width_16k/canonical"
    GENMA_SCOPE_15_WIDTH_16K_CANONICAL = "layer_15/width_16k/canonical"
    GENMA_SCOPE_16_WIDTH_16K_CANONICAL = "layer_16/width_16k/canonical"
    GENMA_SCOPE_17_WIDTH_16K_CANONICAL = "layer_17/width_16k/canonical"
    GENMA_SCOPE_18_WIDTH_16K_CANONICAL = "layer_18/width_16k/canonical"
    GENMA_SCOPE_19_WIDTH_16K_CANONICAL = "layer_19/width_16k/canonical"
    GENMA_SCOPE_20_WIDTH_16K_CANONICAL = "layer_20/width_16k/canonical"
    GENMA_SCOPE_21_WIDTH_16K_CANONICAL = "layer_21/width_16k/canonical"
    GENMA_SCOPE_22_WIDTH_16K_CANONICAL = "layer_22/width_16k/canonical"
    GENMA_SCOPE_23_WIDTH_16K_CANONICAL = "layer_23/width_16k/canonical"
    GENMA_SCOPE_24_WIDTH_16K_CANONICAL = "layer_24/width_16k/canonical"
    LLAMA_SCOPE_0_WIDTH_32k = "l0r_32x"
    LLAMA_SCOPE_1_WIDTH_32k = "l1r_32x"
    LLAMA_SCOPE_2_WIDTH_32k = "l2r_32x"
    LLAMA_SCOPE_3_WIDTH_32k = "l3r_32x"
    LLAMA_SCOPE_4_WIDTH_32k = "l4r_32x"
    LLAMA_SCOPE_5_WIDTH_32k = "l5r_32x"
    LLAMA_SCOPE_6_WIDTH_32k = "l6r_32x"
    LLAMA_SCOPE_7_WIDTH_32k = "l7r_32x"
    LLAMA_SCOPE_8_WIDTH_32k = "l8r_32x"
    LLAMA_SCOPE_9_WIDTH_32k = "l9r_32x"
    LLAMA_SCOPE_10_WIDTH_32k = "l10r_32x"
    LLAMA_SCOPE_11_WIDTH_32k = "l11r_32x"
    LLAMA_SCOPE_12_WIDTH_32k = "l12r_32x"
    LLAMA_SCOPE_13_WIDTH_32k = "l13r_32x"
    LLAMA_SCOPE_14_WIDTH_32k = "l14r_32x"
    LLAMA_SCOPE_15_WIDTH_32k = "l15r_32x"
    LLAMA_SCOPE_16_WIDTH_32k = "l16r_32x"
    LLAMA_SCOPE_17_WIDTH_32k = "l17r_32x"
    LLAMA_SCOPE_18_WIDTH_32k = "l18r_32x"
    LLAMA_SCOPE_19_WIDTH_32k = "l19r_32x"
    LLAMA_SCOPE_20_WIDTH_32k = "l20r_32x"
    LLAMA_SCOPE_21_WIDTH_32k = "l21r_32x"
    LLAMA_SCOPE_22_WIDTH_32k = "l22r_32x"
    LLAMA_SCOPE_23_WIDTH_32k = "l23r_32x"
    LLAMA_SCOPE_24_WIDTH_32k = "l24r_32x"
    LLAMA_SCOPE_25_WIDTH_32k = "l25r_32x"
    LLAMA_SCOPE_26_WIDTH_32k = "l26r_32x"
    LLAMA_SCOPE_27_WIDTH_32k = "l27r_32x"
    LLAMA_SCOPE_28_WIDTH_32k = "l28r_32x"
    LLAMA_SCOPE_29_WIDTH_32k = "l29r_32x"
    LLAMA_SCOPE_30_WIDTH_32k = "l30r_32x"
    LLAMA_SCOPE_31_WIDTH_32k = "l31r_32x"


class CachedComponent(str, Enum):
    HOOK_SAE_ACTS_POST = "hook_resid_pre.hook_sae_acts_post"
    HOOK_RESID_SAE_ACTS_POST = "hook_resid_post.hook_sae_acts_post"


@dataclass
class SAEAnalysis:
    model: AvailableModels
    logit_lens_logit_diffs: list
    labels: list
    plot_type: PlotType
    index: list
    release: str
    sae_id: str
    cache_component: str


def main(
    model_arg: str,
    csv_path: str,
    release: str,
    sae_id: str,
    layer: int,
    cache_component: str,
    output_dir: str,
    before_func_col: str = "func_before",
    after_func_col: str = "func_after",
):
    MSR_df = pd.read_csv(csv_path)
    model = HookedSAETransformer.from_pretrained(model_arg, device=device)
    logger.info("Loading Model...")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    logger.info("Model loaded")

    for i in trange(len(MSR_df)):
        import math

        LIMIT = math.inf
        prompt = [str(MSR_df.iloc[i][after_func_col])]
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > LIMIT:
            print("Skiping")
            continue

        prompt = [str(MSR_df.iloc[i][before_func_col])]
        tokens = model.to_tokens(prompt, prepend_bos=True)
        if tokens.shape[1] > LIMIT:
            print("Skiping")
            continue

        _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])
        index = [f"feature_{i}" for i in range(sae.cfg.d_sae)]

        feature_activation_df = pd.DataFrame(
            cache["blocks" + "." + str(layer) + "." + cache_component][0, -1, :]
            .cpu()
            .numpy(),
            index=index,
        )
        feature_activation_df.columns = ["vulnerable"]

        prompt = [str(MSR_df.iloc[i][after_func_col])]

        _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])
        index = [f"feature_{i}" for i in range(sae.cfg.d_sae)]

        feature_activation_df["secure"] = (
            cache["blocks" + "." + str(layer) + "." + cache_component][0, -1, :]
            .cpu()
            .numpy()
        )
        feature_activation_df["diff"] = abs(
            feature_activation_df["vulnerable"] - feature_activation_df["secure"]
        )

        safe_values = feature_activation_df["secure"].values
        vuln_values = feature_activation_df["vulnerable"].values
        diff_values = feature_activation_df["diff"].values

        sae_analysis_safe = SAEAnalysis(
            model=model_arg,
            logit_lens_logit_diffs=safe_values.tolist(),
            labels=index,
            plot_type=PlotType.SAE_FEATURE_IMPORTANCE,
            index=i,
            release=release,
            sae_id=sae_id,
            cache_component=cache_component,
        )
        store_values(
            os.path.join(output_dir, "feature_importance_safe.jsonl"),
            append=True,
            index=sae_analysis_safe.index,
            model=sae_analysis_safe.model,
            plot_type=sae_analysis_safe.plot_type,
            sae_id=sae_analysis_safe.sae_id,
            cache_component=sae_analysis_safe.cache_component,
            values=sae_analysis_safe.logit_lens_logit_diffs,
            labels=sae_analysis_safe.labels,
        )

        sae_analysis_vuln = SAEAnalysis(
            model=model_arg,
            logit_lens_logit_diffs=vuln_values.tolist(),
            labels=index,
            plot_type=PlotType.SAE_FEATURE_IMPORTANCE,
            index=i,
            release=release,
            sae_id=sae_id,
            cache_component=cache_component,
        )

        store_values(
            os.path.join(output_dir, "feature_importance_vuln.jsonl"),
            append=True,
            index=sae_analysis_vuln.index,
            model=sae_analysis_vuln.model,
            plot_type=sae_analysis_vuln.plot_type,
            sae_id=sae_analysis_vuln.sae_id,
            cache_component=sae_analysis_vuln.cache_component,
            values=sae_analysis_vuln.logit_lens_logit_diffs,
            labels=sae_analysis_vuln.labels,
        )

        sae_analysis_diff = SAEAnalysis(
            model=model_arg,
            logit_lens_logit_diffs=diff_values.tolist(),
            labels=index,
            plot_type=PlotType.SAE_FEATURE_IMPORTANCE,
            index=i,
            release=release,
            sae_id=sae_id,
            cache_component=cache_component,
        )

        store_values(
            os.path.join(output_dir, "feature_importance_diff.jsonl"),
            append=True,
            index=sae_analysis_diff.index,
            model=sae_analysis_diff.model,
            plot_type=sae_analysis_diff.plot_type,
            sae_id=sae_analysis_diff.sae_id,
            cache_component=sae_analysis_diff.cache_component,
            values=sae_analysis_diff.logit_lens_logit_diffs,
            labels=sae_analysis_diff.labels,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_options = [
        model.value for model in list(AvailableModels.__members__.values())
    ]
    parser.add_argument(
        "--model",
        type=AvailableModels,
        help="The model to analyse",
        choices=model_options,
        default=AvailableModels.GPT2_SMALL.value,
    )

    parser.add_argument(
        "--release",
        type=Release,
        help="The release to analyse",
        choices=[r.value for r in list(Release.__members__.values())],
        default=Release.GPT2_SMALL_RES_JB.value,
    )

    parser.add_argument(
        "--sae_id",
        type=SAE_ID,
        help="The SAE ID to analyse",
        choices=[s.value for s in list(SAE_ID.__members__.values())],
        default=SAE_ID.BLOCKS_0_HOOK_RESID_PRE.value,
    )

    parser.add_argument(
        "--cache_component",
        type=CachedComponent,
        help="The cache component to analyse",
        choices=[c.value for c in list(CachedComponent.__members__.values())],
        default=CachedComponent.HOOK_SAE_ACTS_POST.value,
    )

    parser.add_argument(
        "--layer",
        type=int,
        help="The layer to analyse",
    )

    parser.add_argument("--csv_path")

    parser.add_argument(
        "--before_func_col",
        type=str,
        default="func_before",
        help="The column name for the function before the change",
    )

    parser.add_argument(
        "--after_func_col",
        type=str,
        default="func_after",
        help="The column name for the function after the change",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts",
        help="The directory to store the output files",
    )

    args = parser.parse_args()
    main(
        model_arg=args.model,
        csv_path=args.csv_path,
        release=args.release,
        sae_id=args.sae_id,
        layer=args.layer,
        cache_component=args.cache_component,
        before_func_col=args.before_func_col,
        after_func_col=args.after_func_col,
        output_dir=args.output_dir,
    )

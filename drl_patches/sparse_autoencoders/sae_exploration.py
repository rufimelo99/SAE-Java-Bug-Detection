import argparse
from dataclasses import dataclass
from enum import Enum
import os
import pandas as pd
import torch
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm, trange

from drl_patches.logger import logger
from drl_patches.sparse_autoencoders.analyse_layers import store_values
from drl_patches.sparse_autoencoders.schemas import AvailableModels, PlotType

torch.set_grad_enabled(False)
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("Getting device.", device=device)


class Release(str, Enum):
    GPT2_SMALL_RES_JB = "gpt2-small-res-jb"


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


class CachedComponent(str, Enum):
    HOOK_SAE_ACTS_POST = "hook_sae_acts_post"


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
    cache_component: str,
    output_dir: str,
    before_func_col: str = "func_before",
    after_func_col: str = "func_after",
):
    MSR_df = pd.read_csv(csv_path)
    model = HookedSAETransformer.from_pretrained(model_arg, device=device)

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )

    for i in trange(len(MSR_df)):
        prompt = [
            str(MSR_df.iloc[i][before_func_col]),
            str(MSR_df.iloc[i][after_func_col]),
        ]
        _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])

        index = [f"feature_{i}" for i in range(sae.cfg.d_sae)]
        feature_activation_df = pd.DataFrame(
            cache[sae_id + "." + cache_component][0, -1, :].cpu().numpy(),
            index=index,
        )
        feature_activation_df.columns = ["vulnerable"]
        feature_activation_df["secure"] = (
            cache[sae_id + "." + cache_component][1, -1, :].cpu().numpy()
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
        "--csv_path",
        default="artifacts/gbug-java.csv",
    )

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
        cache_component=args.cache_component,
        before_func_col=args.before_func_col,
        after_func_col=args.after_func_col,
        output_dir=args.output_dir,
    )

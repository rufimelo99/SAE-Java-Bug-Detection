import argparse
import json
from typing import Any

import einops
import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from tqdm import tqdm
from transformer_lens import HookedTransformer

from drl_patches.logger import logger
from drl_patches.sparse_autoencoders.schemas import AvailableModels, PlotType
from drl_patches.sparse_autoencoders.utils import (
    imshow,
    line,
    residual_stack_to_logit_diff,
    scatter,
    visualize_attention_patterns,
)

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
logger.info("Device", device=DEVICE)

from dataclasses import dataclass


@dataclass
class LayerAnalysis:
    model: AvailableModels
    logit_lens_logit_diffs: list
    labels: list
    plot_type: PlotType
    index: list


def store_values(
    jsonl_path: str,
    append: bool = False,
    **kwargs: Any,
):
    """
    Stores provided keyword arguments as a JSON object in a .jsonl file.

    Args:
        jsonl_path (str): Path to the JSONL file.
        append (bool): If True, append to the file; otherwise, overwrite. Defaults to False.
        **kwargs (Any): Arbitrary keyword arguments to be stored in the JSON object.
    """
    with open(jsonl_path, "a" if append else "w") as f:
        json.dump(kwargs, f)
        f.write("\n")


def inference(
    model_arg,
    model,
    vulnerable_func,
    safe_func,
    index,
    output_acc_residual_path,
    output_logit_diff_path,
    output_attention_path,
    visualize_figures=False,
):
    # prompts = [MSR_df.iloc[0]["func_before"]] + [MSR_df.iloc[0]["func_after"]]
    prompts = [vulnerable_func] + [safe_func]

    tokens = model.to_tokens(prompts, prepend_bos=True)
    # Run the model and cache all activations.
    original_logits: Float[torch.Tensor, "batch seq_len voc_size"]
    original_logits, cache = model.run_with_cache(tokens)

    # This is to understand why the model would prefer 1 over 0. Here we used eos for both approaches. We check the difference between prompts
    # Converts to Token IDs
    logit_diff_directions = torch.tensor(
        [
            model.to_single_token(model.tokenizer.eos_token),
            model.to_single_token(model.tokenizer.eos_token),
        ]
    )
    # Creates Embeddings for both tokens
    logit_diff_directions = model.tokens_to_residual_directions(logit_diff_directions)

    accumulated_residual, labels = cache.accumulated_resid(
        layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
    )
    logit_lens_logit_diffs = residual_stack_to_logit_diff(
        prompts, accumulated_residual, cache, logit_diff_directions
    )
    if visualize_figures:
        fig = line(
            logit_lens_logit_diffs,
            x=np.arange(model.cfg.n_layers * 2 + 1) / 2,
            hover_name=labels,
            title="Logit Difference From Accumulate Residual Stream for dataset example {}".format(
                index
            ),
        )
        # Save the figure
        # fig.write_html("logit_lens_logit_diffs.html")

    # TODO: This seems repetitive. I just added so we know the structure of the data
    accumulated_res_la = LayerAnalysis(
        model=model_arg,
        logit_lens_logit_diffs=logit_lens_logit_diffs.tolist(),
        labels=labels,
        plot_type=PlotType.ACCUMULATED_RESIDUAL,
        index=index,
    )

    store_values(
        output_acc_residual_path,
        append=True,
        index=accumulated_res_la.index,
        model=accumulated_res_la.model.value,
        logit_lens_logit_diffs=accumulated_res_la.logit_lens_logit_diffs,
        labels=accumulated_res_la.labels,
        plot_type=accumulated_res_la.plot_type,
    )

    per_layer_residual, labels = cache.decompose_resid(
        layer=-1, pos_slice=-1, return_labels=True
    )
    per_layer_logit_diffs = residual_stack_to_logit_diff(
        prompts, per_layer_residual, cache, logit_diff_directions
    )

    logit_diff_layer_la = LayerAnalysis(
        model=model_arg,
        logit_lens_logit_diffs=per_layer_logit_diffs.tolist(),
        labels=labels,
        plot_type=PlotType.LAYER_WISE,
        index=index,
    )

    store_values(
        output_acc_residual_path,
        append=True,
        index=logit_diff_layer_la.index,
        model=logit_diff_layer_la.model.value,
        logit_lens_logit_diffs=logit_diff_layer_la.logit_lens_logit_diffs,
        labels=logit_diff_layer_la.labels,
        plot_type=logit_diff_layer_la.plot_type,
    )

    if visualize_figures:
        fig = line(
            per_layer_logit_diffs,
            hover_name=labels,
            title="Logit Difference From Each Layer for dataset example {}".format(
                index
            ),
        )
        # fig.write_html("per_layer_logit_diffs.html")
    per_head_residual, labels = cache.stack_head_results(
        layer=-1, pos_slice=-1, return_labels=True
    )
    per_head_logit_diffs = residual_stack_to_logit_diff(
        prompts, per_head_residual, cache, logit_diff_directions
    )
    per_head_logit_diffs = einops.rearrange(
        per_head_logit_diffs,
        "(layer head_index) -> layer head_index",
        layer=model.cfg.n_layers,
        head_index=model.cfg.n_heads,
    )

    logit_diff_head_la = LayerAnalysis(
        model=model_arg,
        logit_lens_logit_diffs=per_head_logit_diffs.tolist(),
        labels=labels,
        plot_type=PlotType.SAE_FEATURE_IMPORTANCE,
        index=index,
    )

    store_values(
        output_acc_residual_path,
        append=True,
        index=logit_diff_head_la.index,
        model=logit_diff_head_la.model.value,
        logit_lens_logit_diffs=logit_diff_head_la.logit_lens_logit_diffs,
        labels=logit_diff_head_la.labels,
        plot_type=logit_diff_head_la.plot_type,
    )

    if visualize_figures:
        imshow(
            per_head_logit_diffs,
            labels={"x": "Head", "y": "Layer"},
            title="Logit Difference From Each Head",
        )


def main(
    model_arg: AvailableModels,
    output_acc_residual_path: str = "artifacts/accumulated_residual_stream.jsonl",
    output_logit_diff_path: str = "artifacts/logit_difference_by_layer.jsonl",
    output_attention_path: str = "artifacts/attention_patterns.jsonl",
    before_func_col: str = "func_before",
    after_func_col: str = "func_after",
):
    logger.info("Loading Model.", model=model_arg)

    model = HookedTransformer.from_pretrained(
        model_arg.value,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        # refactor_factored_attn_matrices=True,
        device=DEVICE,
    )
    torch.set_grad_enabled(False)
    print("Disabled automatic differentiation")

    for index, row in tqdm(MSR_df.iterrows(), total=MSR_df.shape[0]):
        inference(
            model_arg,
            model,
            row[before_func_col],
            row[after_func_col],
            index,
            output_acc_residual_path,
            output_logit_diff_path,
            output_attention_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse the layers of a model through mechanistic Interpretability"
    )

    model_options = [
        model.value for model in list(AvailableModels.__members__.values())
    ]
    parser.add_argument(
        "--model",
        type=AvailableModels,
        help="The model to analyse",
        choices=model_options,
        default=AvailableModels.GPT2_SMALL.value,
        required=True,
    )
    parser.add_argument(
        "--csv_path",
        default="artifacts/MSR_data_cleaned_vul.csv",
    )

    args = parser.parse_args()

    # TODO: Fix this for other entries
    MSR_df = pd.read_csv(args.csv_path)
    args.model = AvailableModels(args.model)
    main(model_arg=args.model)

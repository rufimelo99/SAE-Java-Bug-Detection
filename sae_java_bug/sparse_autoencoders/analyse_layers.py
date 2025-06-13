import argparse
import json
import os
from typing import Any

import einops
import numpy as np
import pandas as pd
import torch
from sae_java_bug.logger import logger
from sae_java_bug.sparse_autoencoders.schemas import AvailableModels, PlotType
from sae_java_bug.sparse_autoencoders.utils import (
    imshow,
    line,
    residual_stack_to_logit_diff,
    scatter,
    visualize_attention_patterns,
)
from jaxtyping import Float
from tqdm import tqdm
from transformer_lens import HookedTransformer

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
    # If there is a directory path in the JSONL path, create the directory
    directory = "/".join(jsonl_path.split("/")[:-1])
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(jsonl_path, "a" if append else "w") as f:
        json.dump(kwargs, f)
        f.write("\n")


def inference(
    model_arg: AvailableModels,
    model: HookedTransformer,
    vulnerable_func: str,
    safe_func: str,
    index: int,
    output_acc_residual_path: str,
    output_logit_diff_path: str,
    output_attention_path: str,
    visualize_figures: bool = False,
):
    prompts = [str(vulnerable_func), str(safe_func)]
    # prompts = [str(vulnerable_func)]

    # Tokenize the prompts
    tokens = model.to_tokens(prompts)

    # Run the model with caching for both prompts
    _, cache = model.run_with_cache(tokens)

    accumulated_residual, labels = cache.accumulated_resid(
        layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
    )

    scaled_residual_stack = cache.apply_ln_to_stack(
        accumulated_residual, layer=-1, pos_slice=-1
    )

    # torch.Size([25, 2, 768])

    scaled_residual_stack_ = (
        scaled_residual_stack[:, 0, :] - scaled_residual_stack[:, 1, :]
    )
    scaled_residual_stack_diff = torch.linalg.vector_norm(
        scaled_residual_stack_, dim=1, ord=2
    )

    logit_diff_layer_la = LayerAnalysis(
        model=model_arg,
        logit_lens_logit_diffs=scaled_residual_stack_diff.tolist(),
        labels=labels,
        plot_type=PlotType.LAYER_WISE,
        index=index,
    )

    store_values(
        output_logit_diff_path,
        append=True,
        index=logit_diff_layer_la.index,
        model=logit_diff_layer_la.model.value,
        values=logit_diff_layer_la.logit_lens_logit_diffs,
        labels=logit_diff_layer_la.labels,
        plot_type=logit_diff_layer_la.plot_type,
    )

    if visualize_figures:
        fig = line(
            scaled_residual_stack_diff,
            hover_name=labels,
            title="Logit Difference From Each Layer for dataset example {}".format(
                index
            ),
        )
        # fig.write_html("per_layer_logit_diffs.html")

    # num_layers = model.cfg.n_layers
    # num_heads = model.cfg.n_heads

    # # Initialize a list to store the average attention differences
    # avg_attention_diffs = []

    # for layer in range(num_layers):
    #     # Extract attention patterns for both prompts
    #     attn_pattern_1 = cache["pattern", layer][
    #         0
    #     ]  # Shape: [num_heads, seq_len, seq_len]
    #     attn_pattern_2 = cache["pattern", layer][
    #         1
    #     ]  # Shape: [num_heads, seq_len, seq_len]

    #     # Compute the absolute difference between the two attention patterns
    #     attn_diff = torch.abs(
    #         attn_pattern_1 - attn_pattern_2
    #     )  # Shape: [num_heads, seq_len, seq_len]

    #     # Average over the sequence length dimensions to get a single value per head
    #     avg_attn_diff_per_head = attn_diff.mean(dim=(1, 2))  # Shape: [num_heads]

    #     avg_attention_diffs.append(avg_attn_diff_per_head)

    # # Convert the list to a tensor for easier manipulation
    # avg_attention_diffs = torch.stack(
    #     avg_attention_diffs
    # )  # Shape: [num_layers, num_heads]

    # logit_diff_head_la = LayerAnalysis(
    #     model=model_arg,
    #     logit_lens_logit_diffs=avg_attention_diffs.tolist(),
    #     labels=labels,
    #     plot_type=PlotType.ATTENTION,
    #     index=index,
    # )

    # store_values(
    #     output_attention_path,
    #     append=True,
    #     index=logit_diff_head_la.index,
    #     model=logit_diff_head_la.model.value,
    #     values=logit_diff_head_la.logit_lens_logit_diffs,
    #     labels=logit_diff_head_la.labels,
    #     plot_type=logit_diff_head_la.plot_type,
    # )

    # if visualize_figures:
    #     imshow(
    #         avg_attention_diffs,
    #         labels={"x": "Head", "y": "Layer"},
    #         title="Logit Difference From Each Head",
    #     )


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
    model.eval()
    torch.set_grad_enabled(False)
    print("Disabled automatic differentiation")

    indexes_to_skip = [6, 12]

    for index, row in tqdm(MSR_df.iterrows(), total=MSR_df.shape[0]):
        if index in indexes_to_skip:
            continue
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

import argparse
import json

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from tqdm import tqdm
from transformer_lens import HookedTransformer

from drl_patches.logger import logger
from drl_patches.sparse_autoencoders.schemas import AvailableModels
from drl_patches.sparse_autoencoders.utils import (
    imshow,
    line,
    residual_stack_to_logit_diff,
    scatter,
    visualize_attention_patterns,
)

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

logger.info("Device", device=device)

# TODO: Fix this for other entries
MSR_df = pd.read_csv("MSR_data_vul.csv")
MSR_df.head()


def store_values(
    jsonl_path: str,
    index: int,
    model: str,
    logit_lens_logit_diffs: np.ndarray,
    labels: list,
    append: bool = False,
):
    with open(jsonl_path, "a" if append else "w") as f:
        json.dump(
            {
                "index": index,
                "logit_diff": logit_lens_logit_diffs.tolist(),
                "labels": labels,
                "model": model,
            },
            f,
        )
        f.write("\n")


def inference(
    model_arg,
    model,
    vulnerable_func,
    safe_func,
    index,
    output_logit_diff_path,
    output_attention_path,
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
    fig = line(
        logit_lens_logit_diffs,
        x=np.arange(model.cfg.n_layers * 2 + 1) / 2,
        hover_name=labels,
        title="Logit Difference From Accumulate Residual Stream for dataset example {}".format(
            index
        ),
    )

    store_values(
        output_logit_diff_path,
        index,
        model_arg.value,
        logit_lens_logit_diffs,
        labels,
        append=True,
    )

    # Save the figure
    # fig.write_html("logit_lens_logit_diffs.html")

    per_layer_residual, labels = cache.decompose_resid(
        layer=-1, pos_slice=-1, return_labels=True
    )
    per_layer_logit_diffs = residual_stack_to_logit_diff(
        prompts, per_layer_residual, cache, logit_diff_directions
    )

    store_values(
        output_attention_path,
        index,
        model_arg.value,
        per_layer_logit_diffs,
        labels,
        append=True,
    )

    fig = line(
        per_layer_logit_diffs,
        hover_name=labels,
        title="Logit Difference From Each Layer for dataset example {}".format(index),
    )
    # fig.write_html("per_layer_logit_diffs.html")


def main(
    model_arg: AvailableModels,
    output_logit_diff_path: str = "accumulated_residual_stream.jsonl",
    output_attention_path: str = "logit_difference_by_layer.jsonl",
):
    logger.info("Loading Model.", model=model_arg)

    model = HookedTransformer.from_pretrained(
        model_arg.value,
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    torch.set_grad_enabled(False)
    print("Disabled automatic differentiation")

    for index, row in tqdm(MSR_df.iterrows(), total=MSR_df.shape[0]):
        inference(
            model_arg,
            model,
            row["func_before"],
            row["func_after"],
            index,
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

    args = parser.parse_args()
    args.model = AvailableModels(args.model)
    main(model_arg=args.model)

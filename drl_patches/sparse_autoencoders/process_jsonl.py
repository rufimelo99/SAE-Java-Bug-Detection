import argparse
import json

import numpy as np
import torch
from transformer_lens import HookedTransformer

from drl_patches.logger import logger
from drl_patches.sparse_autoencoders.utils import line

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse the layers of a model through mechanistic Interpretability"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    n_lines = 0
    avgs = {}
    with open(args.input, "r") as f:
        for fline in f:
            line_content = json.loads(fline)
            if n_lines == 0:
                avgs["logit_diff"] = line_content["logit_diff"]
                avgs["labels"] = line_content["labels"]
                avgs["model"] = line_content["model"]

            else:
                for i, logit_diff in enumerate(line_content["logit_diff"]):
                    avgs["logit_diff"][i] += logit_diff

            n_lines += 1

    for i, logit_diff in enumerate(avgs["logit_diff"]):
        avgs["logit_diff"][i] /= n_lines

    model = HookedTransformer.from_pretrained(
        avgs["model"],
        center_unembed=True,
        center_writing_weights=True,
        fold_ln=True,
        refactor_factored_attn_matrices=True,
    )
    torch.set_grad_enabled(False)
    print("Disabled automatic differentiation")

    fig = line(
        avgs["logit_diff"],
        hover_name=avgs["labels"],
        title="Average Logit Difference From Accumulate Residual Stream",
    )

    with open(args.output, "w") as f:
        json.dump(avgs, f)
        f.write("\n")

    logger.info("Done")

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from fancy_einsum import einsum
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sae_java_bug.logger import logger

if torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu"
logger.info("Getting device.", device=DEVICE)

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
random.seed(seed)


def write_jsonl(data: json, file_path, append=False):
    mode = "a" if append else "w"

    if not os.path.exists(file_path):
        mode = "w"

    with open(file_path, mode) as f:
        f.write(json.dumps(data) + "\n")


# Function to get hidden states
def get_hidden_states(tokenizer, model, text: str):
    """
    Get the hidden states from the model for a given text.

    Args:
        text (str): Input text.

    Returns:
        Tuple[torch.Tensor]: Hidden states from all layers.
    """
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.eos_token = tokenizer.pad_token
    tokenizer.eos_token_id = tokenizer.pad_token_id
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = (
            outputs.hidden_states
        )  # Tuple of (layer_num, batch_size, seq_len, hidden_dim)
    return hidden_states


def main(model_name: str, csv_path: str, output_dir: str):
    """
    Main function to train the model.

    Args:
        model_name (str): Name of the pre-trained model.
        csv_path (str): Path to the CSV file containing the data.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Load the data
    df = pd.read_csv(csv_path)
    logger.info("Data loaded.")

    # Preprocess the data
    # func_before and func_after are the columns that we care about
    df["func_before"] = df["func_before"].apply(lambda x: x.replace(" ", ""))
    df["func_after"] = df["func_after"].apply(lambda x: x.replace(" ", ""))

    # Get hidden states for each row in the DataFrame
    for index, row in tqdm(df.iterrows(), total=len(df)):
        func_before = row["func_before"]
        func_after = row["func_after"]

        if isinstance(func_before, str) and isinstance(func_after, str):
            hidden_states_before = get_hidden_states(tokenizer, model, func_before)
            hidden_states_after = get_hidden_states(tokenizer, model, func_after)
        else:
            logger.warning("Skipping row due to non-string values.")
            continue
        if hidden_states_before and hidden_states_after:
            for hidden_state_idx, hidden_state in enumerate(hidden_states_before):
                hidden_state = einsum("batch seq dim -> dim", hidden_state)
                hidden_state = hidden_state.cpu().numpy().tolist()
                layer_dir = os.path.join(output_dir, f"layer{hidden_state_idx}")
                os.makedirs(layer_dir, exist_ok=True)
                write_jsonl(
                    {
                        "func_before": func_before,
                        "values": hidden_state,
                        "layer": hidden_state_idx,
                        "model": model_name,
                        "vuln": 0,
                        "labels": np.arange(len(hidden_state)).tolist(),
                        "plot_type": "something",
                    },
                    os.path.join(layer_dir, "feature_importance_safe.jsonl"),
                    append=True,
                )

            for hidden_state_idx, hidden_state in enumerate(hidden_states_after):
                hidden_state = einsum("batch seq dim -> dim", hidden_state)
                hidden_state = hidden_state.cpu().numpy().tolist()
                layer_dir = os.path.join(output_dir, f"layer{hidden_state_idx}")
                os.makedirs(layer_dir, exist_ok=True)

                write_jsonl(
                    {
                        "func_after": func_after,
                        "values": hidden_state,
                        "layer": hidden_state_idx,
                        "model": model_name,
                        "vuln": 1,
                        "labels": np.arange(len(hidden_state)).tolist(),
                        "plot_type": "something",
                    },
                    os.path.join(layer_dir, "feature_importance_vuln.jsonl"),
                    append=True,
                )

            # Compute the diff between the two hidden states
            for hidden_state_idx, (
                hidden_state_before,
                hidden_state_after,
            ) in enumerate(zip(hidden_states_before, hidden_states_after)):
                hidden_state_before = einsum(
                    "batch seq dim -> dim", hidden_state_before
                )
                hidden_state_after = einsum("batch seq dim -> dim", hidden_state_after)

                # Compute the diff
                diff = torch.abs(hidden_state_before - hidden_state_after)
                diff = diff.cpu().numpy().tolist()

                layer_dir = os.path.join(output_dir, f"layer{hidden_state_idx}")
                os.makedirs(layer_dir, exist_ok=True)

                write_jsonl(
                    {
                        "values": diff,
                        "layer": hidden_state_idx,
                        "model": model_name,
                        "labels": np.arange(len(diff)).tolist(),
                        "plot_type": "something",
                    },
                    os.path.join(layer_dir, "feature_importance_diff.jsonl"),
                    append=True,
                )

        else:
            logger.warning("Skipping row due to empty hidden states.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a modern BERT model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Name of the pre-trained model.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to the CSV file containing the data.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="hidden_states",
        help="Directory to save the model and tokenizer.",
    )

    args = parser.parse_args()

    main(args.model_name, args.csv_path, args.output_dir)

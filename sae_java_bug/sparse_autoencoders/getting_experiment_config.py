import argparse
import json
import random
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch

from sae_java_bug.logger import logger

warnings.filterwarnings("ignore")

# Set up seeds.
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
random.seed(seed)


def load_training_indexes(config_path: str) -> List[int]:
    # Loads the training indexes from a jsonl file.
    with open(config_path, "r") as f:
        training_indexes = json.load(f)
    return training_indexes


def get_training_indexes(diff_df):
    return np.random.choice(diff_df.index, int(len(diff_df) * 0.8), replace=False)


def save_config(
    training_indexes: List[int],
    config_path: str,
):
    # Saves the training indexes to a jsonl file.
    with open(config_path, "w") as f:
        json.dump(training_indexes, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv_path",
        type=str,
        default="artifacts/defects4j.csv",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="artifacts/defects4j_train_indexes.jsonl",
    )

    args = parser.parse_args()

    logger.info("Reading data.")
    diff_df = pd.read_csv(args.csv_path)

    train_indexes = get_training_indexes(diff_df)
    logger.info(f"Training indexes: {train_indexes}")

    logger.info("Saving config.")
    save_config(
        training_indexes=train_indexes.tolist(),
        config_path=args.output_path,
    )

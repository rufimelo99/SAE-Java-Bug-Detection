import argparse
import json
import os
import warnings

import pandas as pd
from sae_java_bug.logger import logger
from sae_java_bug.sparse_autoencoders.getting_experiment_config import (
    load_training_indexes,
    save_config,
)
from sae_java_bug.sparse_autoencoders.utils import read_jsonl_file, set_seed
from tqdm import trange

warnings.filterwarnings("ignore")

set_seed(42)


def get_diff_data(diff_jsonl_path):
    diff_data = list(read_jsonl_file(diff_jsonl_path))
    diff_df = pd.DataFrame(diff_data)

    columns = diff_df.columns.to_list()
    columns.remove("values")

    diff_df.drop(columns=columns, inplace=True)

    for i in trange(len(diff_df["values"][0])):
        diff_df[f"feature_{i}"] = diff_df["values"].apply(lambda x: x[i])

    diff_df.drop(columns=["values"], inplace=True)

    return diff_df


def sort_features(train_df_diff):
    return list(train_df_diff.sum(axis=0).sort_values(ascending=False).index)


def load_sorted_features(config_path: str):
    # Loads the sorted features from a jsonl file.
    with open(config_path, "r") as f:
        sorted_features = json.load(f)
    return sorted_features


def get_top_features(sorted_features, n=100):
    return sorted_features[:n]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir-path",
        type=str,
        default="artifacts",
    )

    parser.add_argument(
        "--train-indexes_path",
        type=str,
        default="artifacts/gbug-java_train_indexes.json",
    )

    args = parser.parse_args()

    logger.info("Reading data.")
    diff_df = get_diff_data(
        os.path.join(args.dir_path, "feature_importance_diff.jsonl")
    )
    train_indexes = load_training_indexes(args.train_indexes_path)
    train_df_diff = diff_df.loc[train_indexes]
    most_important_cols = sort_features(train_df_diff)

    save_config(
        most_important_cols,
        os.path.join(args.dir_path, "most_important_features.json"),
    )

    logger.info("Saved most important features.")

import argparse
import os
import pickle
from typing import List

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from sae_java_bug.logger import logger
from sae_java_bug.sparse_autoencoders.utils import set_seed

tqdm.pandas()
torch.set_grad_enabled(False)
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
logger.info("Getting device.", device=device)
set_seed(42)


def create_tfidf_vectorizer(
    dfs: List[pd.DataFrame],
    before_func_col: str,
    after_func_col: str,
    output_dir: str,
    max_features: int = 5000,
) -> TfidfVectorizer:

    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        lowercase=True,
        token_pattern=r"\b\w+\b",
        stop_words="english",
    )
    # Fit the vectorizer on the data from the before and after columns
    text = []
    for df in dfs:
        text += df[before_func_col].tolist() + df[after_func_col].tolist()
    vectorizer.fit(text)

    vectorizer_path = os.path.join(output_dir, "vectorizer.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    return vectorizer


def load_tfidf_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    # Load the TF-IDF vectorizer from a file
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer


def main(
    csvs: List[str],
    output_dir: str,
    before_func_col: str = "func_before",
    after_func_col: str = "func_after",
):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the data
    logger.info("Loading data.")
    dfs = []
    for other_csv in csvs:
        other_df = pd.read_csv(other_csv)
        dfs.append(other_df)

    logger.info("Training vectorizer.")
    vectorizer = create_tfidf_vectorizer(
        dfs, before_func_col, after_func_col, output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csvs",
        type=str,
        nargs="+",  # Accept multiple values as a list
        default=[
            "artifacts/gbug-java.csv",
            "artifacts/defects4j.csv",
            "artifacts/humaneval.csv",
        ],  # Default as a list
        help="The paths to the other CSV files (space-separated if multiple)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts",
        help="The directory to store the output files",
    )

    args = parser.parse_args()
    main(
        args.csvs,
        args.output_dir,
    )

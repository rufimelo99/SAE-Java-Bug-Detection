import argparse
import os
import pickle

import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

from sae_java_bug.logger import logger
from sae_java_bug.sparse_autoencoders.analyse_layers import store_values
from sae_java_bug.sparse_autoencoders.get_vectorizer import load_tfidf_vectorizer
from sae_java_bug.sparse_autoencoders.getting_experiment_config import (
    load_training_indexes,
)
from sae_java_bug.sparse_autoencoders.schemas import AvailableModels
from sae_java_bug.sparse_autoencoders.utils import set_seed
from sae_java_bug.sparse_autoencoders.vulnerability_detection_features import (
    ClassifierType,
    parameters_map,
    sk_classifiers_map,
)

tqdm.pandas()
torch.set_grad_enabled(False)
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
logger.info("Getting device.", device=device)
set_seed(42)


def get_metrics(y_pred, y_test):
    # Initialize counts
    TP = FP = TN = FN = 0

    for pred, actual in zip(y_pred, y_test):
        if pred == 1 and actual == 1:
            TP += 1
        elif pred == 1 and actual == 0:
            FP += 1
        elif pred == 0 and actual == 0:
            TN += 1
        elif pred == 0 and actual == 1:
            FN += 1

    # Calculate metrics
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    return precision, recall, accuracy, f1


def main(
    csv_path: str,
    vectorizer_path: str,
    output_dir: str,
    before_func_col: str = "func_before",
    after_func_col: str = "func_after",
    train_indexes_path: str = "artifacts/gbug-java_train_indexes.json",
):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the data
    df = pd.read_csv(csv_path)
    vectorizer = load_tfidf_vectorizer(vectorizer_path)
    logger.info("Vectorizer loaded.")

    logger.info("Data loaded.")

    df["tokenized_before"] = df[before_func_col].progress_apply(
        lambda x: vectorizer.transform([x]).toarray()[0]
    )
    df["tokenized_after"] = df[after_func_col].progress_apply(
        lambda x: vectorizer.transform([x]).toarray()[0]
    )
    # Pad to 5000 tokens
    df["tokenized_before"] = df["tokenized_before"].apply(
        lambda x: x[:5000] + [0] * (5000 - len(x)) if len(x) < 5000 else x[:5000]
    )
    df["tokenized_after"] = df["tokenized_after"].apply(
        lambda x: x[:5000] + [0] * (5000 - len(x)) if len(x) < 5000 else x[:5000]
    )

    logger.info("Tokenization done.")

    train_indexes = load_training_indexes(train_indexes_path)
    df_train = df.loc[train_indexes]
    df_test = df.drop(train_indexes)

    df_classical_train = pd.DataFrame()
    df_classical_test = pd.DataFrame()
    for row in df_train.iterrows():
        row = row[1]
        df_classical_train = pd.concat(
            [
                df_classical_train,
                pd.DataFrame(
                    {"tokens": [row["tokenized_before"].tolist()], "vuln": 1}, index=[0]
                ),
            ]
        )
        df_classical_train = pd.concat(
            [
                df_classical_train,
                pd.DataFrame(
                    {"tokens": [row["tokenized_after"].tolist()], "vuln": 0}, index=[0]
                ),
            ]
        )

    for row in df_test.iterrows():
        row = row[1]
        df_classical_test = pd.concat(
            [
                df_classical_test,
                pd.DataFrame(
                    {"tokens": [row["tokenized_before"].tolist()], "vuln": 1}, index=[0]
                ),
            ]
        )
        df_classical_test = pd.concat(
            [
                df_classical_test,
                pd.DataFrame(
                    {"tokens": [row["tokenized_after"].tolist()], "vuln": 0}, index=[0]
                ),
            ]
        )

    df_classical_train.reset_index(drop=True, inplace=True)
    df_classical_test.reset_index(drop=True, inplace=True)

    # Shuffle the data
    df_classical_train = df_classical_train.sample(frac=1).reset_index(drop=True)
    df_classical_test = df_classical_test.sample(frac=1).reset_index(drop=True)

    X_train = df_classical_train["tokens"].values.tolist()
    X_train = [torch.tensor(x) for x in X_train]
    y_train = df_classical_train["vuln"]

    X_test = df_classical_test["tokens"].values.tolist()
    X_test = [torch.tensor(x) for x in X_test]
    y_test = df_classical_test["vuln"]

    # KNN
    model = ClassifierType.KNN

    logger.warning("Limiting n_jobs to 2 to avoid memory issues in the cluster.")
    clf = GridSearchCV(
        sk_classifiers_map[model],
        parameters_map[model],
        cv=5,
        verbose=2,
        n_jobs=2,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    precision, recall, accuracy, f1 = get_metrics(y_pred, y_test)
    logger.info(
        "Classification report:",
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        f1=f1,
    )
    store_values(
        os.path.join(output_dir, "classifier_info.jsonl"),
        append=True,
        model="KNN",
        n_features=5000,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        f1=f1,
        y_pred=y_pred.tolist(),
        y_test=y_test.tolist(),
        params=clf.best_params_,
        dataset=csv_path,
    )
    # Save the model
    model_name = f"{model.value}_k_{5000}.pt"
    with open(os.path.join(output_dir, model_name), "wb") as f:
        pickle.dump(clf, f)
    logger.info("Model saved.", model_name=model_name, output_dir=output_dir)

    # Random Forest
    model = ClassifierType.RANDOM_FOREST
    logger.warning("Limiting n_jobs to 2 to avoid memory issues in the cluster.")
    clf = GridSearchCV(
        sk_classifiers_map[model],
        parameters_map[model],
        cv=5,
        verbose=2,
        n_jobs=2,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision, recall, accuracy, f1 = get_metrics(y_pred, y_test)
    logger.info(
        "Classification report:",
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        f1=f1,
    )
    store_values(
        os.path.join(output_dir, "classifier_info.jsonl"),
        append=True,
        model="RandomForest",
        n_features=5000,
        precision=precision,
        recall=recall,
        accuracy=accuracy,
        f1=f1,
        y_pred=y_pred.tolist(),
        y_test=y_test.tolist(),
        params=clf.best_params_,
        dataset=csv_path,
    )

    # Save the model
    model_name = f"{model.value}_k_{5000}.pt"
    with open(os.path.join(output_dir, model_name), "wb") as f:
        pickle.dump(clf, f)
    logger.info("Model saved.", model_name=model_name, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_options = [
        model.value for model in list(AvailableModels.__members__.values())
    ]

    parser.add_argument("--csv_path")

    parser.add_argument(
        "--vectorizer_path",
        type=str,
        default="artifacts/vectorizer.pkl",
        help="The path to the vectorizer file",
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
        "--train-indexes_path",
        type=str,
        default="artifacts/gbug-java_train_indexes.json",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts",
        help="The directory to store the output files",
    )

    args = parser.parse_args()
    main(
        args.csv_path,
        args.vectorizer_path,
        args.output_dir,
        args.before_func_col,
        args.after_func_col,
        args.train_indexes_path,
    )

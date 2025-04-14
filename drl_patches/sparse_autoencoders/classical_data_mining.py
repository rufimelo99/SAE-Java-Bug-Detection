import argparse
import os
import random
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
import torch
from drl_patches.logger import logger
from drl_patches.sparse_autoencoders.analyse_layers import store_values
from drl_patches.sparse_autoencoders.schemas import AvailableModels, PlotType
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Classical logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Do grid search
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tqdm import tqdm, trange
from transformers import AutoTokenizer

tqdm.pandas()
torch.set_grad_enabled(False)
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
logger.info("Getting device.", device=device)
# Set up seeds
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
random.seed(seed)

OUR_STANDARD_TOKENIZER = "meta-llama/Llama-3.1-8B"
from transformers import AutoTokenizer


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


def get_training_indexes(diff_df):
    return np.random.choice(diff_df.index, int(len(diff_df) * 0.8), replace=False)


def main(
    csv_path: str,
    output_dir: str,
    before_func_col: str = "func_before",
    after_func_col: str = "func_after",
):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(OUR_STANDARD_TOKENIZER)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # Load the data
    df = pd.read_csv(csv_path)
    logger.info("Data loaded.")

    df["tokenized_before"] = df["func_before"].progress_apply(
        lambda x: tokenizer(
            x,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=5000,
        )["input_ids"]
    )
    df["tokenized_after"] = df["func_after"].progress_apply(
        lambda x: tokenizer(
            x,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=5000,
        )["input_ids"]
    )

    train_indexes = get_training_indexes(df)
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
                    {"tokens": row["tokenized_before"].tolist(), "vuln": 1}, index=[0]
                ),
            ]
        )
        df_classical_train = pd.concat(
            [
                df_classical_train,
                pd.DataFrame(
                    {"tokens": row["tokenized_after"].tolist(), "vuln": 0}, index=[0]
                ),
            ]
        )

    for row in df_test.iterrows():
        row = row[1]
        df_classical_test = pd.concat(
            [
                df_classical_test,
                pd.DataFrame(
                    {"tokens": row["tokenized_before"].tolist(), "vuln": 1}, index=[0]
                ),
            ]
        )
        df_classical_test = pd.concat(
            [
                df_classical_test,
                pd.DataFrame(
                    {"tokens": row["tokenized_after"].tolist(), "vuln": 0}, index=[0]
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

    # param_grid = {
    #     "C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
    #     "max_iter": [1000],
    #     "solver": ["newton-cg", "lbfgs"],
    # }

    # logger.info(
    #     "Starting grid search.", classifier="LogisticRegression", param_grid=param_grid
    # )
    # clf = GridSearchCV(
    #     LogisticRegression(random_state=0), param_grid, cv=5, verbose=3
    # )  # Verbose level 3 for detailed output

    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)

    # precision, recall, accuracy, f1 = get_metrics(y_pred, y_test)

    # logger.info(
    #     "Classification report:",
    #     precision=precision,
    #     recall=recall,
    #     accuracy=accuracy,
    #     f1=f1,
    # )
    # store_values(
    #     os.path.join(output_dir, "classifier_info.jsonl"),
    #     append=True,
    #     model="LogisticRegression",
    #     n_features=5000,
    #     precision=precision,
    #     recall=recall,
    #     accuracy=accuracy,
    #     f1=f1,
    #     y_pred=y_pred.tolist(),
    #     y_test=y_test.tolist(),
    #     params=clf.best_params_,
    # )

    # logger.info(
    #     "Finished grid search.", classifier="LogisticRegression", param_grid=param_grid
    # )
    # # PCA Regression

    # pca = PCA()
    # clf = LogisticRegression()
    # pipe = Pipeline(steps=[("pca", pca), ("logistic", clf)])

    # param_grid_pca = {
    #     "pca__n_components": [5, 50, 100, 500],
    #     "logistic__C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #     "logistic__max_iter": [1000],
    #     "logistic__solver": ["newton-cg", "lbfgs"],
    # }

    # logger.info(
    #     "Starting grid search.",
    #     classifier="PCA + LogisticRegression",
    #     param_grid=param_grid_pca,
    # )

    # clf = GridSearchCV(
    #     pipe, param_grid_pca, cv=5, verbose=3
    # )  # Verbose level 3 for detailed output

    # clf.fit(X_train, y_train)

    # y_pred = clf.predict(X_test)

    # precision, recall, accuracy, f1 = get_metrics(y_pred, y_test)
    # logger.info(
    #     "Classification report:",
    #     precision=precision,
    #     recall=recall,
    #     accuracy=accuracy,
    #     f1=f1,
    # )

    # store_values(
    #     os.path.join(output_dir, "classifier_info.jsonl"),
    #     append=True,
    #     model="PCA + LogisticRegression",
    #     n_features=5000,
    #     precision=precision,
    #     recall=recall,
    #     accuracy=accuracy,
    #     f1=f1,
    #     y_pred=y_pred.tolist(),
    #     y_test=y_test.tolist(),
    #     params=clf.best_params_,
    # )

    # logger.info(
    #     "Finished grid search.",
    #     classifier="PCA + LogisticRegression",
    #     param_grid=param_grid_pca,
    # )

    # KNN

    param_grid_knn = {
        "n_neighbors": [1, 3, 5, 7, 10, 20, 50],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean"],
    }

    logger.info("Starting grid search.", classifier="KNN", param_grid=param_grid_knn)

    clf = GridSearchCV(
        KNeighborsClassifier(), param_grid_knn, cv=5, verbose=3
    )  # Verbose level 3 for detailed output

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

    # logger.info("Finished grid search.", classifier="KNN", param_grid=param_grid_knn)

    # Random Forest
    # param_grid_rf = {
    #     "n_estimators": [100, 300, 1000],
    #     "max_features": ["sqrt", "log2"],
    #     "max_depth": [None, 10, 50, 100],
    #     "min_samples_split": [2, 5, 10],
    #     "min_samples_leaf": [1, 2, 4],
    # }
    # logger.info(
    #     "Starting grid search.", classifier="RandomForest", param_grid=param_grid_rf
    # )

    # clf = GridSearchCV(
    #     RandomForestClassifier(), param_grid_rf, cv=5, verbose=3
    # )  # Verbose level 3 for detailed output

    # clf.fit(X_train, y_train)

    # y_pred = clf.predict(X_test)
    # precision, recall, accuracy, f1 = get_metrics(y_pred, y_test)
    # logger.info(
    #     "Classification report:",
    #     precision=precision,
    #     recall=recall,
    #     accuracy=accuracy,
    #     f1=f1,
    # )
    # store_values(
    #     os.path.join(output_dir, "classifier_info.jsonl"),
    #     append=True,
    #     model="RandomForest",
    #     n_features=5000,
    #     precision=precision,
    #     recall=recall,
    #     accuracy=accuracy,
    #     f1=f1,
    #     y_pred=y_pred.tolist(),
    #     y_test=y_test.tolist(),
    #     params=clf.best_params_,
    #     dataset=csv_path,
    # )

    # logger.info(
    #     "Finished grid search.", classifier="RandomForest", param_grid=param_grid_rf
    # )

    # SVM

    # param_grid_svm = {
    #     "C": [0.001, 0.01, 0.1, 1, 10, 100],
    #     "kernel": ["linear", "rbf"],
    #     "gamma": ["scale", 0.01, 0.1, 1],  # only applies to rbf
    # }

    # logger.info("Starting grid search.", classifier="SVM", param_grid=param_grid_svm)

    # clf = GridSearchCV(
    #     SVC(), param_grid_svm, cv=5, verbose=3
    # )  # Verbose level 3 for detailed output

    # clf.fit(X_train, y_train)

    # y_pred = clf.predict(X_test)
    # precision, recall, accuracy, f1 = get_metrics(y_pred, y_test)
    # logger.info(
    #     "Classification report:",
    #     precision=precision,
    #     recall=recall,
    #     accuracy=accuracy,
    #     f1=f1,
    # )
    # store_values(
    #     os.path.join(output_dir, "classifier_info.jsonl"),
    #     append=True,
    #     model="SVM",
    #     n_features=5000,
    #     precision=precision,
    #     recall=recall,
    #     accuracy=accuracy,
    #     f1=f1,
    #     y_pred=y_pred.tolist(),
    #     y_test=y_test.tolist(),
    #     params=clf.best_params_,
    # )

    # logger.info("Finished grid search.", classifier="SVM", param_grid=param_grid_svm)

    # # Neural Network

    # param_grid_nn = {
    #     "hidden_layer_sizes": [(100,), (100, 100), (100, 100, 100)],
    #     "activation": ["tanh", "relu"],
    #     "solver": ["adam"],
    #     "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
    #     "learning_rate": ["constant"],
    # }

    # logger.info(
    #     "Starting grid search.", classifier="NeuralNetwork", param_grid=param_grid_nn
    # )

    # clf = GridSearchCV(
    #     MLPClassifier(), param_grid_nn, cv=5, verbose=3
    # )  # Verbose level 3 for detailed output

    # clf.fit(X_train, y_train)

    # y_pred = clf.predict(X_test)

    # precision, recall, accuracy, f1 = get_metrics(y_pred, y_test)
    # logger.info(
    #     "Classification report:",
    #     precision=precision,
    #     recall=recall,
    #     accuracy=accuracy,
    #     f1=f1,
    # )
    # store_values(
    #     os.path.join(output_dir, "classifier_info.jsonl"),
    #     append=True,
    #     model="MLP",
    #     n_features=5000,
    #     precision=precision,
    #     recall=recall,
    #     accuracy=accuracy,
    #     f1=f1,
    #     y_pred=y_pred.tolist(),
    #     y_test=y_test.tolist(),
    #     params=clf.best_params_,
    # )

    # logger.info(
    #     "Finished grid search.", classifier="NeuralNetwork", param_grid=param_grid_nn
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    model_options = [
        model.value for model in list(AvailableModels.__members__.values())
    ]

    parser.add_argument("--csv_path")

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
        args.csv_path,
        args.output_dir,
        args.before_func_col,
        args.after_func_col,
    )

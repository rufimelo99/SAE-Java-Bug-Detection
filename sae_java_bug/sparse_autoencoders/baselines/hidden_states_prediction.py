import argparse
import json
import os
import random
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm, trange

from sae_java_bug.logger import logger
from sae_java_bug.sparse_autoencoders.vulnerability_detection_features import (
    ClassifierType,
    parameters_map,
    read_jsonl_file,
    sk_classifiers_map,
    store_classifier_info,
)

warnings.filterwarnings("ignore")


# Set up seeds
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
random.seed(seed)


def get_training_indexes(diff_df):
    return np.random.choice(diff_df.index, int(len(diff_df) * 0.8), replace=False)


def get_vuln_safe_data(vuln_jsonl_path, safe_jsonl_path):
    vuln_data = list(read_jsonl_file(vuln_jsonl_path))
    safe_data = list(read_jsonl_file(safe_jsonl_path))
    vuln_df = pd.DataFrame(vuln_data)
    safe_df = pd.DataFrame(safe_data)
    vuln_df.drop(columns=["labels", "model", "plot_type"], inplace=True)
    vuln_df["vuln"] = 1

    safe_df.drop(columns=["labels", "model", "plot_type"], inplace=True)
    safe_df["vuln"] = 0

    for i in trange(len(vuln_df["values"][0])):
        vuln_df[f"feature_{i}"] = vuln_df["values"].apply(lambda x: x[i])
        safe_df[f"feature_{i}"] = safe_df["values"].apply(lambda x: x[i])

    safe_df_train = safe_df.loc[train_indexes]
    safe_df_test = safe_df.drop(train_indexes)

    vuln_df_train = vuln_df.loc[train_indexes]
    vuln_df_test = vuln_df.drop(train_indexes)

    df_train = pd.concat([safe_df_train, vuln_df_train])
    df_test = pd.concat([safe_df_test, vuln_df_test])

    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)
    df_train.drop(columns=["values"], inplace=True)
    df_test.drop(columns=["values"], inplace=True)

    return df_train, df_test


def get_most_important_features(train_df_diff, n=100):
    return train_df_diff.sum(axis=0).sort_values(ascending=False).index[1 : n + 1]


def train_model(
    df_train,
    df_test,
    top_k_features,
    sk_classifiers: List[ClassifierType],
    directory="artifacts",
    column="hidden_state",
):

    X_train = df_train[column]
    X_train = pd.DataFrame(
        [np.array(x) for x in X_train],
        columns=[f"feature_{i}" for i in range(len(X_train[0]))],
    )
    y_train = df_train["vuln"]

    X_test = df_test[column]
    X_test = pd.DataFrame(
        [np.array(x) for x in X_test],
        columns=[f"feature_{i}" for i in range(len(X_test[0]))],
    )
    y_test = df_test["vuln"]

    for clf in sk_classifiers:
        classifier = sk_classifiers_map[clf]
        classifier = GridSearchCV(
            classifier, param_grid=parameters_map[clf], cv=5, verbose=3
        )

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        tp = sum((y_pred == 1) & (y_test == 1))
        fp = sum((y_pred == 1) & (y_test == 0))
        tn = sum((y_pred == 0) & (y_test == 0))
        fn = sum((y_pred == 0) & (y_test == 1))
        store_classifier_info(
            directory + f"/{clf.value}.jsonl",
            clf,
            top_k_features,
            tp,
            fp,
            tn,
            fn,
            y_pred,
            y_test,
            classifier.best_params_,
            directory,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dir-path",
        type=str,
        default="gpt2_hidden_states_defects/layer0/",
    )

    args = parser.parse_args()

    logger.info("Reading data.")

    hidden_states_after_df = pd.read_json(
        os.path.join(args.dir_path, "hidden_states_after.jsonl"),
        lines=True,
    )

    hidden_states_before_df = pd.read_json(
        os.path.join(args.dir_path, "hidden_states_before.jsonl"),
        lines=True,
    )

    train_indexes = get_training_indexes(hidden_states_after_df)
    train_df_after = hidden_states_after_df.loc[train_indexes]
    train_df_before = hidden_states_before_df.loc[train_indexes]

    test_df_after = hidden_states_after_df.drop(train_indexes)
    test_df_before = hidden_states_before_df.drop(train_indexes)

    df_train = pd.concat([train_df_after, train_df_before])
    df_test = pd.concat([test_df_after, test_df_before])
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    df_test = df_test.sample(frac=1).reset_index(drop=True)

    logger.info("Training models.")
    top_k_features = [
        1,
        2,
        3,
        4,
        5,
        10,
        20,
        25,
        50,
        100,
        200,
        500,
        1000,
        2000,
        5000,
        10000,
    ]
    for k in tqdm(top_k_features):
        train_model(
            df_train,
            df_test,
            k,
            [
                # ClassifierType.LOGISTIC_REGRESSION,
                # ClassifierType.SVM,
                # ClassifierType.KNN,
                # ClassifierType.DECISION_TREE,
                ClassifierType.RANDOM_FOREST,
            ],
            args.dir_path,
        )

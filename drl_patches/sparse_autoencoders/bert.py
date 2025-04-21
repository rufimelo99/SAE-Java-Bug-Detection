import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from drl_patches.logger import logger
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm, trange
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

random.seed(42)
tqdm.pandas()


class LossLoggerCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.eval_loss = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                self.train_loss.append((state.global_step, logs["loss"]))
            if "eval_loss" in logs:
                self.eval_loss.append((state.global_step, logs["eval_loss"]))


class DisableWandbCallback(TrainerCallback):
    def on_init_end(self, args, state, control, **kwargs):
        import os

        os.environ["WANDB_MODE"] = "disabled"


def train_bert_model(dataset_path, training_indices_path, model_name):
    logger.info("Training BERT model", dataset_path=dataset_path)
    MODEL_NAME = model_name
    logger.info("Model name", model_name=MODEL_NAME)

    logger.info("Reading data.")

    # Load the training indices
    with open(training_indices_path, "r") as f:
        training_indices = json.load(f)

    df = pd.read_csv(dataset_path)
    temp_train = df.iloc[training_indices]
    temp_test = df.drop(training_indices)
    # bug_id func_before func_after
    # Duplicate the rows for func_before with vuln=1 and func_after with vuln=0

    df_train = pd.DataFrame(columns=["bug_id", "function", "vuln"])
    for index, row in temp_train.iterrows():
        df_train = pd.concat(
            [
                df_train,
                pd.DataFrame(
                    {
                        "bug_id": [row["bug_id"]],
                        "function": [row["func_before"]],
                        "vuln": [1],
                    }
                ),
            ]
        )
        df_train = pd.concat(
            [
                df_train,
                pd.DataFrame(
                    {
                        "bug_id": [row["bug_id"]],
                        "function": [row["func_after"]],
                        "vuln": [0],
                    }
                ),
            ]
        )
    df_train = df_train.reset_index(drop=True)

    df_test = pd.DataFrame(columns=["bug_id", "function", "vuln"])
    for index, row in temp_test.iterrows():
        df_test = pd.concat(
            [
                df_test,
                pd.DataFrame(
                    {
                        "bug_id": [row["bug_id"]],
                        "function": [row["func_before"]],
                        "vuln": [1],
                    }
                ),
            ]
        )
        df_test = pd.concat(
            [
                df_test,
                pd.DataFrame(
                    {
                        "bug_id": [row["bug_id"]],
                        "function": [row["func_after"]],
                        "vuln": [0],
                    }
                ),
            ]
        )
    df_test = df_test.reset_index(drop=True)

    # Shuffle
    df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

    # Preprocess the data
    train_data = df_train["function"].tolist()
    train_labels = df_train["vuln"].tolist()
    test_data = df_test["function"].tolist()
    test_labels = df_test["vuln"].tolist()
    train_data = [str(code) for code in train_data]
    test_data = [str(code) for code in test_data]
    # Tokenize the data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(
        train_data, truncation=True, padding=True, max_length=512
    )
    test_encodings = tokenizer(test_data, truncation=True, padding=True, max_length=512)

    # Create a dataset class
    class CodeDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=1)
        f1 = f1_score(
            labels, predictions, average="weighted"
        )  # or "macro"/"micro" depending on your task
        return {"f1": f1}

    train_dataset = CodeDataset(train_encodings, train_labels)
    test_dataset = CodeDataset(test_encodings, test_labels)

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    # Let Trainer handle the device placement
    logger.info("Model loaded. GPU available", yes=torch.cuda.is_available())

    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_dir="./logs",
        load_best_model_at_end=True,
        learning_rate=2e-5,
        optim="adamw_torch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        metric_for_best_model="f1",  # <---- this tells Trainer what metric to track
        greater_is_better=True,  # <---- F1 is a score where higher is better
    )

    loss_logger = LossLoggerCallback()
    disable_wandb = DisableWandbCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[loss_logger, disable_wandb],
    )

    logger.info("Starting training...")
    trainer.train()

    # Evaluate the model
    evaluation = trainer.evaluate()
    logger.info("Evaluation completed", evaluation=evaluation)
    train_losses = loss_logger.train_loss
    eval_losses = loss_logger.eval_loss

    # store the losses
    train_loss_df = pd.DataFrame(train_losses, columns=["step", "loss"])
    eval_loss_df = pd.DataFrame(eval_losses, columns=["step", "loss"])
    train_loss_df.to_csv(
        f"{dataset_path}_train_losses.csv".replace("/", "_"), index=False
    )
    eval_loss_df.to_csv(
        f"{dataset_path}_eval_losses.csv".replace("/", "_"), index=False
    )

    # Get the best model and save it
    best_model = trainer.model
    best_model.save_pretrained(
        f"{MODEL_NAME}_{dataset_path}_best_model".replace("/", "_")
    )  # Save the model
    tokenizer.save_pretrained(
        f"{MODEL_NAME}_{dataset_path}_best_model".replace("/", "_")
    )  # Save the tokenizer

    # Evaluate again with the best model
    trainer = Trainer(
        model=best_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[loss_logger, disable_wandb],
    )
    logger.info("Starting evaluation...")
    evaluation = trainer.evaluate()
    logger.info("Evaluation completed", evaluation=evaluation)
    # Save the evaluation results
    eval_df = pd.DataFrame([evaluation], columns=evaluation.keys())
    eval_df.to_csv(f"{dataset_path}_eval_results.csv".replace("/", "_"), index=False)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    accuracy = accuracy_score(labels, predictions)

    logger.info(
        "Metrics",
        accuracy=accuracy,
        f1=f1,
        precision=precision,
        recall=recall,
    )

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="artifacts/gbug-java.csv",
        help="csv file with the dataset",
    )

    parser.add_argument(
        "--training_indices",
        type=str,
        default="artifacts/gbug-java_train_indexes.json",
        help="json file with the training indices",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="model name",
        choices=[
            "microsoft/graphcodebert-base",
            "answerdotai/ModernBERT-base",
            "answerdotai/ModernBERT-large",
        ],
    )

    args = parser.parse_args()

    # Load the dataset
    dataset_path = args.dataset
    training_indices_path = args.training_indices
    # Load the model
    model_name = args.model
    train_bert_model(dataset_path, training_indices_path, model_name)

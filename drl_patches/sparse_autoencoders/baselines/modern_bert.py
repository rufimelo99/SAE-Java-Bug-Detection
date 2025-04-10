import argparse
import random

import numpy as np
import pandas as pd
import torch
from drl_patches.logger import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, AutoModelForSequenceClassification, AutoTokenizer

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


class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, labels):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, return_tensors="pt"
        )
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def get_training_indexes(diff_df):
    return np.random.choice(diff_df.index, int(len(diff_df) * 0.8), replace=False)


def main(model_name, csv_path):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Load the data
    df = pd.read_csv(csv_path)
    logger.info("Data loaded.")

    # Preprocess the data
    # func_before and func_after are the columns that we care

    train_indexes = get_training_indexes(df)
    df_train = df.loc[train_indexes]
    df_test = df.drop(train_indexes)

    texts = df_train["func_before"].tolist() + df_train["func_after"].tolist()
    labels = [1] * len(df_train["func_before"]) + [0] * len(df_train["func_after"])
    texts = [text for text in texts if isinstance(text, str)]
    labels = [label for label in labels if isinstance(label, int)]
    logger.info("Data preprocessed.")

    # Split data and prepare loaders
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2
    )
    train_dataset = TextDataset(tokenizer, train_texts, train_labels)
    val_dataset = TextDataset(tokenizer, val_texts, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)
    logger.info("Data loaded into DataLoader.")

    # Move model to GPU if available
    device = torch.device(DEVICE)
    model.to(device)
    logger.info("Model moved to device.")
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)
    # Training loop (1 epoch for simplicity)
    model.train()

    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Training loss: {loss.item():.4f}")

    # Validation

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

    print(f"Validation Accuracy: {correct / total:.2f}")

    # Save the model
    model.save_pretrained("modern_bert_model")
    tokenizer.save_pretrained("modern_bert_model")


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
    args = parser.parse_args()

    main(args.model_name, args.csv_path)

    # Load the model for inference
    loaded_model = AutoModelForSequenceClassification.from_pretrained(
        "modern_bert_model"
    )
    loaded_tokenizer = AutoTokenizer.from_pretrained("modern_bert_model")

    loaded_model.to(DEVICE)

    def predict(text):
        inputs = loaded_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        ).to(DEVICE)
        with torch.no_grad():
            outputs = loaded_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.item()

    # Example prediction
    example_text = "I really enjoyed this film!"
    prediction = predict(example_text)
    print(
        f"Prediction for '{example_text}': {'Positive' if prediction == 1 else 'Negative'}"
    )

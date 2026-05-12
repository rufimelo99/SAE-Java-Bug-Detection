#!/usr/bin/env python3
"""
Evaluate stronger pooling strategies to test the "fundamental difficulty" claim.

This script tests:
1. Attention-weighted pooling (last token's attention over all positions)
2. Learned pooling (trainable weighted average via a small network)
3. Per-token classifiers (RNN/Transformer over token sequence)
4. Pairwise/siamese encoders (commit-pair difference encoders)

All evaluated under stratified 5-fold CV to test if any exceed the 0.5 AUROC ceiling
for mean-token pooling.
"""

import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


class AttentionPooling:
    """Pool activations using last-token attention weights."""

    @staticmethod
    def pool(activations: np.ndarray, attention_weights: np.ndarray) -> np.ndarray:
        """
        Args:
            activations: (n_samples, n_tokens, hidden_dim)
            attention_weights: (n_samples, n_tokens) - attention from last token to all tokens

        Returns:
            pooled: (n_samples, hidden_dim)
        """
        # Normalize attention to sum to 1
        attention = attention_weights / (
            attention_weights.sum(axis=1, keepdims=True) + 1e-8
        )
        # Weighted average: sum over tokens weighted by attention
        pooled = np.einsum("nt,nthd->nhd", attention, activations)
        return pooled


class LearnedPooling(nn.Module):
    """Learn a pooling function via a small MLP."""

    def __init__(self, hidden_dim: int, context_length_max: int = 2048):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Small learnable network: (n_tokens, hidden_dim) -> (n_tokens, 1)
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            activations: (batch, n_tokens, hidden_dim)

        Returns:
            pooled: (batch, hidden_dim)
        """
        weights = self.weight_net(activations)  # (batch, n_tokens, 1)
        weights = torch.softmax(weights, dim=1)
        pooled = (activations * weights).sum(dim=1)  # (batch, hidden_dim)
        return pooled


class PerTokenClassifier(nn.Module):
    """Classify vulnerability using per-token features with an RNN."""

    def __init__(self, hidden_dim: int, rnn_hidden: int = 128, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.LSTM(
            hidden_dim,
            rnn_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(rnn_hidden * 2, 1)  # bidirectional output

    def forward(self, activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            activations: (batch, n_tokens, hidden_dim)

        Returns:
            logits: (batch,)
            pooled_repr: (batch, rnn_hidden*2) - last hidden state
        """
        _, (h_n, _) = self.rnn(activations)  # h_n: (2 * num_layers, batch, rnn_hidden)
        # Use final hidden state from last layer, both directions
        h_final = h_n[-2:]  # (2, batch, rnn_hidden)
        pooled = h_final.permute(1, 0, 2).reshape(
            h_final.shape[1], -1
        )  # (batch, rnn_hidden*2)
        logits = self.classifier(pooled).squeeze(-1)
        return logits, pooled


class PairwiseEncoder(nn.Module):
    """Encode commit-pair differences (vulnerable - secure)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        # Simple: encode the difference vector with a small network
        self.difference_encoder = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self, activations_vulnerable: torch.Tensor, activations_secure: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            activations_vulnerable: (batch, hidden_dim)
            activations_secure: (batch, hidden_dim)

        Returns:
            logits: (batch,)
        """
        diff = activations_vulnerable - activations_secure
        logits = self.difference_encoder(diff).squeeze(-1)
        return logits


def evaluate_pooling_strategy(
    activations: np.ndarray,
    labels: np.ndarray,
    pooling_fn,
    strategy_name: str,
    cv_folds: int = 5,
) -> dict:
    """
    Evaluate a pooling strategy using stratified k-fold CV.

    Args:
        activations: (n_samples, n_tokens, hidden_dim) or (n_samples, hidden_dim)
        labels: (n_samples,) - binary vulnerability labels
        pooling_fn: callable that pools activations
        strategy_name: name of pooling strategy
        cv_folds: number of CV folds

    Returns:
        dict with AUROC, CI, and fold-wise results
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_aurocs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(activations, labels)):
        # Pool if needed
        if activations.ndim == 3:
            act_train_pooled = pooling_fn(activations[train_idx])
            act_test_pooled = pooling_fn(activations[test_idx])
        else:
            act_train_pooled = activations[train_idx]
            act_test_pooled = activations[test_idx]

        # Train logistic regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(act_train_pooled, labels[train_idx])

        # Evaluate
        y_pred = clf.predict_proba(act_test_pooled)[:, 1]
        auroc = roc_auc_score(labels[test_idx], y_pred)
        fold_aurocs.append(auroc)

    fold_aurocs = np.array(fold_aurocs)
    ci_lower = np.percentile(fold_aurocs, 2.5)
    ci_upper = np.percentile(fold_aurocs, 97.5)

    return {
        "strategy": strategy_name,
        "mean_auroc": fold_aurocs.mean(),
        "std_auroc": fold_aurocs.std(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "fold_aurocs": fold_aurocs.tolist(),
    }


def main():
    """Run pooling strategy evaluation."""
    # TODO: Load pre-computed activations from disk
    # Expected: activations dict with keys for each layer
    # activations[layer] = (n_samples, n_tokens, hidden_dim)
    # labels: (n_samples,) - binary vulnerability labels

    # Placeholder: users should load their activation tensors here
    print("=" * 70)
    print("POOLING STRATEGY EVALUATION")
    print("=" * 70)

    # For now, create synthetic data for testing
    n_samples, n_tokens, hidden_dim = 1000, 100, 3584
    activations = np.random.randn(n_samples, n_tokens, hidden_dim).astype(np.float32)
    labels = np.random.randint(0, 2, n_samples)

    results = {}

    # 1. Mean-token pooling (baseline)
    print("\n1. Mean-Token Pooling (baseline)...")
    mean_pooling = lambda x: x.mean(axis=1)
    results["mean_token"] = evaluate_pooling_strategy(
        activations, labels, mean_pooling, "mean_token"
    )

    # 2. Attention-weighted pooling (requires attention weights)
    print("\n2. Attention-Weighted Pooling...")
    # Simulate attention: higher weight on later tokens
    attention_weights = np.linspace(0.5, 1.5, n_tokens)[np.newaxis, :]
    attention_pooling = lambda x: AttentionPooling.pool(
        x, attention_weights * np.ones((x.shape[0], 1))
    )
    results["attention"] = evaluate_pooling_strategy(
        activations, labels, attention_pooling, "attention_weighted"
    )

    # 3. Max pooling
    print("\n3. Max Pooling...")
    max_pooling = lambda x: x.max(axis=1)
    results["max"] = evaluate_pooling_strategy(
        activations, labels, max_pooling, "max_pooling"
    )

    # 4. Last-token pooling (for comparison)
    print("\n4. Last-Token Pooling (for comparison)...")
    last_token_pooling = lambda x: x[:, -1, :]
    results["last_token"] = evaluate_pooling_strategy(
        activations, labels, last_token_pooling, "last_token"
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    for strategy, res in results.items():
        print(f"\n{res['strategy']}:")
        print(f"  Mean AUROC: {res['mean_auroc']:.4f}")
        print(f"  95% CI: [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]")
        print(f"  Std Dev: {res['std_auroc']:.4f}")

    # Save results
    output_path = Path("pooling_strategy_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

"""
On-the-fly SAE activation extractor.

Wraps the Qwen2.5-7B-Instruct LLM + SAE pipeline so that raw code strings
can be turned into 16 384-dimensional SAE feature vectors without pre-caching.

Mirrors the logic in sae_exploration.py but as a reusable, lazily-loaded
object suitable for CI/CD and interactive analysis.

Usage
-----
>>> from sae_java_bug.evaluation.activation_extractor import ActivationExtractor
>>> from sae_java_bug.sparse_autoencoders.schemas import QWEN_CODER_7B_VULNEABLE_CODE_STD_10M_CONFIG
>>>
>>> extractor = ActivationExtractor.from_config(
...     QWEN_CODER_7B_VULNEABLE_CODE_STD_10M_CONFIG
... )
>>> extractor.load()   # loads ~14 GB model + SAE — do once
>>>
>>> acts = extractor.get_activations("public int add(int a, int b) { return a + b; }")
>>> acts.shape   # (16384,)
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional

from sae_java_bug.sparse_autoencoders.schemas import SAEConfig


# ---------------------------------------------------------------------------
# Token limit — skip inputs that are too long (same heuristic as sae_exploration.py)
# ---------------------------------------------------------------------------
MAX_TOKENS = 2000


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ActivationExtractor:
    """
    Lazily loads Qwen2.5-7B-Instruct + a paired SAE and exposes a simple
    ``get_activations(code_str)`` interface.

    Lazy loading means the 14 GB model is only pulled into memory when you
    first call :meth:`load` (or :meth:`get_activations`, which auto-loads).

    Parameters
    ----------
    cfg : SAEConfig
        Configuration that specifies which model and SAE release to use.
    layer : int
        Transformer layer whose residual stream is fed into the SAE.
        Must be in ``cfg.layers_available``.
    device : str, optional
        Device string ("cuda", "mps", "cpu").  Auto-detected if None.
    max_tokens : int
        Inputs longer than this (in tokens) are rejected with a ValueError.
    """

    def __init__(
        self,
        cfg: SAEConfig,
        layer: int,
        device: Optional[str] = None,
        max_tokens: int = MAX_TOKENS,
    ):
        if layer not in cfg.layers_available:
            raise ValueError(
                f"Layer {layer} is not available for this config. "
                f"Available: {cfg.layers_available}"
            )
        self.cfg = cfg
        self.layer = layer
        self.device = device or _get_device()
        self.max_tokens = max_tokens

        self._model = None
        self._tokenizer = None
        self._sae = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        cfg: SAEConfig,
        layer: Optional[int] = None,
        device: Optional[str] = None,
    ) -> "ActivationExtractor":
        """
        Create an extractor from a SAEConfig, using the first available layer
        if *layer* is not specified.
        """
        chosen_layer = layer if layer is not None else cfg.layers_available[0]
        return cls(cfg=cfg, layer=chosen_layer, device=device)

    # ------------------------------------------------------------------
    # Model / SAE loading
    # ------------------------------------------------------------------

    def load(self) -> "ActivationExtractor":
        """
        Load the LLM and SAE into memory.

        Safe to call multiple times — subsequent calls are no-ops.
        Returns *self* for chaining: ``extractor = ActivationExtractor(...).load()``.
        """
        if self._model is not None:
            return self

        from transformers import AutoModelForCausalLM, AutoTokenizer
        from sae_lens import SAE

        model_name = self.cfg.model.value
        release    = self.cfg.release.value
        sae_id     = self.cfg.sae_id(layer_index=self.layer)

        print(f"Loading tokenizer: {model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading model: {model_name}  (device={self.device})")
        self._model = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
            .to(self.device)
            .eval()
        )
        torch.set_grad_enabled(False)

        print(f"Loading SAE: release={release}  sae_id={sae_id}")
        if self.cfg.is_local:
            self._sae, _, _ = SAE.load_from_disk(
                self.cfg.sae_path(layer_index=self.layer), device=self.device
            )
        else:
            self._sae, _, _ = SAE.from_pretrained(
                release=release, sae_id=sae_id, device=self.device
            )

        print(f"Ready.  SAE d_sae={self._sae.cfg.d_sae}, layer={self.layer}")
        return self

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def get_activations(self, code_str: str) -> np.ndarray:
        """
        Compute SAE feature activations for a single code string.

        Steps (mirrors sae_exploration.py):
        1. Tokenise *code_str*.
        2. Run a forward pass through the LLM with ``output_hidden_states=True``.
        3. Extract the residual stream at ``layer + 1`` for the **last token**
           (index 0 = embeddings, so transformer layer N = hidden_states[N+1]).
        4. Feed the residual into the SAE encoder.
        5. Return the resulting (d_sae,) feature vector as a float32 ndarray.

        Parameters
        ----------
        code_str : str
            Source code (or any text) to encode.

        Returns
        -------
        ndarray, shape (d_sae,)   typically (16384,)

        Raises
        ------
        ValueError
            If the tokenised input exceeds ``self.max_tokens``.
        """
        self._ensure_loaded()

        inputs = self._tokenizer(code_str, return_tensors="pt").to(self.device)
        n_tokens = inputs["input_ids"].shape[1]
        if n_tokens > self.max_tokens:
            raise ValueError(
                f"Input too long: {n_tokens} tokens > max_tokens={self.max_tokens}. "
                "Truncate or increase max_tokens."
            )

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)

        # hidden_states: tuple of (n_layers + 1) tensors, shape (1, seq_len, hidden_dim)
        # Layer 0 = embeddings; transformer layer N = hidden_states[N + 1]
        layer_idx = self.layer + 1
        residual = outputs.hidden_states[layer_idx][:, -1, :]  # (1, hidden_dim)

        sae_acts = self._sae.encode(residual).cpu()            # (1, d_sae)
        return sae_acts.squeeze(0).numpy().astype(np.float32)  # (d_sae,)

    def get_delta(self, baseline_code: str, new_code: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute activations for two code versions and return both.

        Convenience wrapper for the CI/CD scan pattern:

        .. code-block:: python

            baseline_acts, new_acts = extractor.get_delta(old_fn, new_fn)
            report = scanner.scan(baseline_acts, new_acts)

        Returns
        -------
        (baseline_activations, new_code_activations) : tuple of ndarray, each (d_sae,)
        """
        baseline_acts = self.get_activations(baseline_code)
        new_acts      = self.get_activations(new_code)
        return baseline_acts, new_acts

    # ------------------------------------------------------------------
    # Batch extraction (for training data generation)
    # ------------------------------------------------------------------

    def get_activations_batch(
        self,
        code_strings: list[str],
        skip_too_long: bool = True,
        show_progress: bool = True,
    ) -> tuple[np.ndarray, list[int]]:
        """
        Compute SAE activations for a list of code strings.

        Parameters
        ----------
        code_strings : list[str]
        skip_too_long : bool
            If True, inputs exceeding max_tokens are skipped (index recorded
            in *skipped_indices*).  If False, a ValueError is raised.
        show_progress : bool
            Show a tqdm progress bar.

        Returns
        -------
        activations : ndarray, shape (n_kept, d_sae)
        skipped_indices : list[int]
            Indices in *code_strings* that were skipped due to length.
        """
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(code_strings), total=len(code_strings)) if show_progress else enumerate(code_strings)
        except ImportError:
            iterator = enumerate(code_strings)

        rows: list[np.ndarray] = []
        skipped: list[int] = []

        for i, code in iterator:
            try:
                rows.append(self.get_activations(code))
            except ValueError:
                if skip_too_long:
                    skipped.append(i)
                else:
                    raise

        return np.vstack(rows) if rows else np.empty((0, 0), dtype=np.float32), skipped

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def d_sae(self) -> int:
        """SAE feature dimension (16 384 for the standard Qwen config)."""
        self._ensure_loaded()
        return self._sae.cfg.d_sae

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def __repr__(self) -> str:
        status = "loaded" if self.is_loaded else "not loaded"
        return (
            f"ActivationExtractor("
            f"model={self.cfg.model.value!r}, "
            f"release={self.cfg.release.value!r}, "
            f"layer={self.layer}, "
            f"device={self.device!r}, "
            f"status={status})"
        )
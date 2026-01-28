import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import einops
import pandas as pd
import torch
from datasets import load_dataset
from jaxtyping import Float
from pydantic import BaseModel
from sae_lens import SAE
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_java_bug.logger import logger
from sae_java_bug.sparse_autoencoders.schemas import (
    CachedComponent,
    ModelFamily,
    Release,
    SAEConfig,
    GEMMA3_CONFIG,
    KODCODE_LLAMA_3_2_1B_CONFIG,
    KODCODE_CODE_LLAMA_7B_CONFIG,
)


class ActivationsSchema(BaseModel):
    vuln_id: str
    secure_code: str
    vulnerable_code: str
    cwe: str
    file_extension: str
    secure: List[float]
    vulnerable: List[float]
    layer: int
    sae_config: SAEConfig

    def append_to_jsonl(self, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if path.exists() else "w"
        with path.open(mode) as f:
            f.write(json.dumps(self.model_dump()) + "\n")


def normalise(txt: str) -> str:
    return txt.replace("/", " ")


torch.set_grad_enabled(False)
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available() else "cpu"
)
logger.info("Getting device.", device=device)

hf_path = "rufimelo/DeltaSecommits"
before_func_col = "prior_version"
after_func_col = "after_version"
vuln_id_col = "vuln_id"
cwe_col = "cwe"
file_ext_col = "file_extension"
# hf_path = "rufimelo/ETHPy150Open_swapped_operand"
# before_func_col = "buggy_version"
# after_func_col = "correct_version"
# vuln_id_col = "info" # placeholder since no vuln_id in this dataset
# cwe_col = "info" # placeholder since no vuln_id in this dataset
# file_ext_col = "info" # placeholder since no vuln_id in this dataset

output_dir = "../artifacts/activations/"
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(output_dir, f"run_{current_time}/")
logger_filepath = f"../artifacts/logs/sae_exploration_{current_time}.log"

MSR_df = load_dataset(hf_path, split="train").to_pandas()

cfg = KODCODE_CODE_LLAMA_7B_CONFIG

MODEL_ARG = cfg.model.value
RELEASE = cfg.release.value
CACHE_COMPONENT = cfg.cached_component.value


def log_warning(message: str, filepath: str):
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if path.exists() else "w"
    with path.open(mode) as f:
        f.write(message + "\n")


def get_residuals(code_str: str, layer_idx: int, tokenizer, model, device=None):
    """Returns residual activations at the specified layer for the last token.

    Args:
        code_str: Input code string
        layer_idx: Which layer's residual stream to extract (0 = embeddings)
        tokenizer: Hugging Face tokenizer
        model: Hugging Face model
        device: Optional device to move inputs to

    Returns:
        Tensor of shape [1, hidden_dim] - residual for the last token
    """
    if device is None:
        device = next(model.parameters()).device

    inputs = tokenizer(code_str, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # hidden_states is tuple of length (num_layers + 1).
    # Index 0: embeddings, Index 1..N: each transformer layer output
    if layer_idx < 0 or layer_idx >= len(outputs.hidden_states):
        raise ValueError(
            f"layer_idx {layer_idx} out of range. Model has {len(outputs.hidden_states)} hidden states."
        )

    residuals: Float[torch.Tensor, "1 seq_len hidden_dim"] = outputs.hidden_states[
        layer_idx
    ]

    # Returns last token's residual.
    return residuals[:, -1, :]


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ARG)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ARG, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    skipped_vuln_ids = set()

    for layer in tqdm(cfg.layers_available):
        SAE_ID = cfg.sae_id(layer_index=layer)
        if cfg.is_local:
            sae,  cfg_dict, sparsity  = SAE.load_from_disk(cfg.sae_path(layer_index=layer), device=device)
        else:
            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=RELEASE, sae_id=SAE_ID, device=device
            )
        for i in trange(len(MSR_df)):
        #for i in trange(200):  # Limit to 200 samples for faster testing
            secure_code = str(MSR_df.iloc[i][before_func_col])
            vulnerable_code = str(MSR_df.iloc[i][after_func_col])
            cwe = str(MSR_df.iloc[i][cwe_col])
            file_extension = str(MSR_df.iloc[i][file_ext_col])
            vuln_id = str(MSR_df.iloc[i][vuln_id_col])

            if vuln_id in skipped_vuln_ids:
                continue

            max_tokens = max(
                len(tokenizer(secure_code, return_tensors="pt")["input_ids"][0]),
                len(tokenizer(vulnerable_code, return_tensors="pt")["input_ids"][0]),
            )
            if max_tokens > 2000:
                warn_msg = f"Skipping vuln_id {vuln_id} due to max tokens {max_tokens} exceeding limit."
                logger.warning(warn_msg)
                log_warning(warn_msg, logger_filepath)
                skipped_vuln_ids.add(vuln_id)
                continue

            # Extract residuals manually.
            layer_after_embeddings = layer + 1

            resid_secure = get_residuals(
                secure_code, layer_after_embeddings, tokenizer, model
            )
            resid_vuln = get_residuals(
                vulnerable_code, layer_after_embeddings, tokenizer, model
            )

            secure_features: Float[torch.Tensor, "1 d_sae"] = sae.encode(
                resid_secure
            ).cpu()
            secure_features = einops.rearrange(secure_features, "1 d_sae -> d_sae")
            vuln_features: Float[torch.Tensor, "1 d_sae"] = sae.encode(resid_vuln).cpu()
            vuln_features = einops.rearrange(vuln_features, "1 d_sae -> d_sae")

            index = [f"feature_{i}" for i in range(sae.cfg.d_sae)]

            feature_activation_df = pd.DataFrame(
                vuln_features, index=index, columns=["vulnerable"]
            )
            feature_activation_df["secure"] = secure_features

            base64_secure = base64.b64encode(secure_code.encode("utf-8"))
            base64_vuln = base64.b64encode(vulnerable_code.encode("utf-8"))

            activations = ActivationsSchema(
                vuln_id=vuln_id,
                secure_code=base64_secure,
                vulnerable_code=base64_vuln,
                secure=secure_features.tolist(),
                vulnerable=vuln_features.tolist(),
                layer=layer,
                sae_config=cfg,
                cwe=cwe,
                file_extension=file_extension,
            )

            activations.append_to_jsonl(
                f"{output_dir}activations_layer_{layer}_sae_{normalise(SAE_ID)}_component_{normalise(CACHE_COMPONENT)}.jsonl"
            )
            logger.info(
                f"Saved activations for vuln_id {vuln_id} at layer {layer} with SAE {SAE_ID} to disk at `{output_dir}`."
            )
            

            torch.cuda.empty_cache()
    # Store a new file with the config used
    with open(f"{output_dir}sae_config_used.json", "w") as f:
        f.write(json.dumps(cfg.model_dump(), indent=4))

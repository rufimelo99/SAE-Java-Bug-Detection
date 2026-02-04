import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import einops
import torch
from datasets import load_dataset
from jaxtyping import Float
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sae_java_bug.logger import logger
from sae_java_bug.sparse_autoencoders.schemas import (
    QWEN_CODER_7B_SECURE_CODE_STRD_CONFIG,
    SAEConfig,
)


class RawActivationsSchema(BaseModel):
    vuln_id: str
    secure_code: str
    vulnerable_code: str
    cwe: str
    file_extension: str
    secure: List[float]
    vulnerable: List[float]
    layer: int
    sae_config: dict
    component: str

    def append_to_jsonl(self, filepath: str):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if path.exists() else "w"
        with path.open(mode) as f:
            f.write(json.dumps(self.model_dump()) + "\n")


def normalise(txt: str) -> str:
    return txt.replace("/", " ")


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

    if layer_idx < 0 or layer_idx >= len(outputs.hidden_states):
        raise ValueError(
            f"layer_idx {layer_idx} out of range. Model has {len(outputs.hidden_states)} hidden states."
        )

    residuals: Float[torch.Tensor, "1 seq_len hidden_dim"] = outputs.hidden_states[
        layer_idx
    ]

    return residuals[:, -1, :]


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(cfg: SAEConfig):
    torch.set_grad_enabled(False)
    device = get_device()
    logger.info("Getting device.", device=device)

    hf_path = "rufimelo/DeltaSecommits"
    before_func_col = "prior_version"
    after_func_col = "after_version"
    vuln_id_col = "vuln_id"
    cwe_col = "cwe"
    file_ext_col = "file_extension"

    output_dir = "../artifacts/activations/"
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(
        output_dir, f"run_{current_time}_{cfg.release.value}_raw/"
    )
    logger_filepath = f"../artifacts/logs/raw_activations_{current_time}.log"

    dataset_df = load_dataset(hf_path, split="train").to_pandas()

    model_arg = cfg.model.value
    component = "hidden_state_last_token"

    tokenizer = AutoTokenizer.from_pretrained(model_arg)
    model = AutoModelForCausalLM.from_pretrained(
        model_arg, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    skipped_vuln_ids = set()

    for layer in cfg.layers_available:
        for i in range(len(dataset_df)):
            secure_code = str(dataset_df.iloc[i][before_func_col])
            vulnerable_code = str(dataset_df.iloc[i][after_func_col])
            cwe = str(dataset_df.iloc[i][cwe_col])
            file_extension = str(dataset_df.iloc[i][file_ext_col])
            vuln_id = str(dataset_df.iloc[i][vuln_id_col])

            if vuln_id in skipped_vuln_ids:
                continue

            max_tokens = max(
                len(tokenizer(secure_code, return_tensors="pt")["input_ids"][0]),
                len(
                    tokenizer(vulnerable_code, return_tensors="pt")["input_ids"][0]
                ),
            )
            if max_tokens > 2000:
                warn_msg = (
                    f"Skipping vuln_id {vuln_id} due to max tokens {max_tokens} exceeding limit."
                )
                logger.warning(warn_msg)
                log_warning(warn_msg, logger_filepath)
                skipped_vuln_ids.add(vuln_id)
                continue

            layer_after_embeddings = layer + 1

            resid_secure = get_residuals(
                secure_code, layer_after_embeddings, tokenizer, model
            )
            resid_vuln = get_residuals(
                vulnerable_code, layer_after_embeddings, tokenizer, model
            )

            secure_residuals = einops.rearrange(
                resid_secure.detach().float().cpu(), "1 d_model -> d_model"
            )
            vuln_residuals = einops.rearrange(
                resid_vuln.detach().float().cpu(), "1 d_model -> d_model"
            )

            base64_secure = base64.b64encode(secure_code.encode("utf-8"))
            base64_vuln = base64.b64encode(vulnerable_code.encode("utf-8"))

            activations = RawActivationsSchema(
                vuln_id=vuln_id,
                secure_code=base64_secure,
                vulnerable_code=base64_vuln,
                secure=secure_residuals.tolist(),
                vulnerable=vuln_residuals.tolist(),
                layer=layer,
                sae_config=cfg.model_dump(),
                cwe=cwe,
                file_extension=file_extension,
                component=component,
            )

            activations.append_to_jsonl(
                f"{output_dir}activations_layer_{layer}_raw_component_{normalise(component)}.jsonl"
            )
            logger.info(
                "Saved raw activations for vuln_id %s at layer %s to disk at `%s`.",
                vuln_id,
                layer,
                output_dir,
            )

            if device == "cuda":
                torch.cuda.empty_cache()

    with open(f"{output_dir}model_config_used.json", "w") as f:
        f.write(json.dumps(cfg.model_dump(), indent=4))


if __name__ == "__main__":
    run(QWEN_CODER_7B_SECURE_CODE_STRD_CONFIG)

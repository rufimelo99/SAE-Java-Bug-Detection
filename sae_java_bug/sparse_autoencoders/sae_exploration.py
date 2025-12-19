import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import torch
from datasets import load_dataset
from pydantic import BaseModel
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm, trange

from sae_java_bug.logger import logger
from sae_java_bug.sparse_autoencoders.schemas import (
    CachedComponent,
    ModelFamily,
    Release,
    SAEConfig,
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

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if path.exists() else "w"

        # Append new activation to list
        with path.open(mode) as f:
            f.write(json.dumps(self.model_dump()) + "\n")


def normalise(txt: str) -> str:
    # replace \n and /
    return txt.replace("/", " ")


torch.set_grad_enabled(False)
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info("Getting device.", device=device)

hf_path = "rufimelo/DeltaSecommits"
before_func_col = "prior_version"
after_func_col = "after_version"
vuln_id_col = "vuln_id"
cwe_col = "cwe"
file_ext_col = "file_extension"

output_dir = "../artifacts/activations/"

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(output_dir, f"run_{current_time}/")
logger_filepath = f"../artifacts/logs/sae_exploration_{current_time}.log"


MSR_df = load_dataset(hf_path, split="train").to_pandas()
cfg = SAEConfig(
    model=ModelFamily.GEMMA,
    release=Release.GEMMA_SCOPE,
    cached_component=CachedComponent.HOOK_RESID_SAE_ACTS_POST,
    layers_available=[i for i in range(26)],
)

MODEL_ARG = cfg.model.value
RELEASE = cfg.release.value
CACHE_COMPONENT = cfg.cached_component.value


def log_warning(message: str, filepath: str):
    path = Path(filepath)

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if path.exists() else "w"

    # Append new activation to list
    with path.open(mode) as f:
        f.write(message + "\n")


if __name__ == "__main__":
    model = HookedSAETransformer.from_pretrained(MODEL_ARG, device=device)

    skypped_vuln_ids = set()

    for layer in tqdm(cfg.layers_available):
        SAE_ID = cfg.sae_id(layer_index=layer)
        sae, cfg_dict, sparsity = SAE.from_pretrained(
            release=RELEASE, sae_id=SAE_ID, device=device
        )

        for i in trange(len(MSR_df)):
            secure_code = str(MSR_df.iloc[i][before_func_col])
            vulnerable_code = str(MSR_df.iloc[i][after_func_col])
            cwe = str(MSR_df.iloc[i][cwe_col])
            file_extension = str(MSR_df.iloc[i][file_ext_col])
            vuln_id = str(MSR_df.iloc[i][vuln_id_col])

            if vuln_id in skypped_vuln_ids:
                continue

            max_tokens = max(
                model.to_tokens(secure_code, prepend_bos=True).shape[1],
                model.to_tokens(vulnerable_code, prepend_bos=True).shape[1],
            )
            if max_tokens > 2000:
                warn_msg = f"Skipping vuln_id {MSR_df.iloc[i][vuln_id_col]} due to max tokens {max_tokens} exceeding limit."
                logger.warning(warn_msg)
                log_warning(warn_msg, logger_filepath)
                skypped_vuln_ids.add(vuln_id)
                continue

            _, cache = model.run_with_cache_with_saes([secure_code], saes=[sae])
            index = [f"feature_{i}" for i in range(sae.cfg.d_sae)]
            feature_activation_df = pd.DataFrame(
                cache["blocks" + "." + str(layer) + "." + CACHE_COMPONENT][0, -1, :]
                .cpu()
                .numpy(),
                index=index,
            )
            feature_activation_df.columns = ["vulnerable"]
            del cache
            torch.cuda.empty_cache()

            _, cache = model.run_with_cache_with_saes([vulnerable_code], saes=[sae])
            index = [f"feature_{i}" for i in range(sae.cfg.d_sae)]

            feature_activation_df["secure"] = (
                cache["blocks" + "." + str(layer) + "." + CACHE_COMPONENT][0, -1, :]
                .cpu()
                .numpy()
            )
            del cache
            torch.cuda.empty_cache()

            safe_values = feature_activation_df["secure"].values
            vuln_values = feature_activation_df["vulnerable"].values

            base64_secure = base64.b64encode(safe_values.tobytes()).decode("utf-8")
            base64_vulnerable = base64.b64encode(vuln_values.tobytes()).decode("utf-8")

            activations = ActivationsSchema(
                vuln_id=vuln_id,
                secure_code=secure_code,
                vulnerable_code=vulnerable_code,
                secure=safe_values.tolist(),
                vulnerable=vuln_values.tolist(),
                layer=layer,
                sae_config=cfg,
                cwe=cwe,
                file_extension=file_extension,
            )

            activations.append_to_jsonl(
                f"{output_dir}activations_layer_{layer}_sae_{normalise(SAE_ID)}_component_{normalise(CACHE_COMPONENT)}.jsonl"
            )

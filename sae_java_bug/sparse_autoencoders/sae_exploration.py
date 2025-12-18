import json
import os
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


torch.set_grad_enabled(False)
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"
logger.info("Getting device.", device=device)

hf_path = "rufimelo/DeltaSecommits"
before_func_col = "prior_version"
after_func_col = "after_version"
vuln_id_col = "vuln_id"
output_dir = "../artifacts/activations/"


MSR_df = load_dataset(hf_path, split="train").to_pandas()
# Filter only 2 samples
MSR_df = MSR_df.sample(n=2, random_state=42)
cfg = SAEConfig(
    model=ModelFamily.GEMMA,
    release=Release.GEMMA_SCOPE,
    cached_component=CachedComponent.HOOK_RESID_SAE_ACTS_POST,
    layers_available=[i for i in range(26)],
)

print(cfg.sae_id)
MODEL_ARG = cfg.model.value
RELEASE = cfg.release.value
SAE_ID = cfg.sae_id
CACHE_COMPONENT = cfg.cached_component.value


model = HookedSAETransformer.from_pretrained(MODEL_ARG, device=device)
logger.info("Loading Model...")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=RELEASE,
    sae_id=SAE_ID,
    device=device,
)
logger.info("Model loaded")

for layer in tqdm(cfg.layers_available):
    for i in trange(len(MSR_df)):
        secure_code = str(MSR_df.iloc[i][before_func_col])
        vulnerable_code = str(MSR_df.iloc[i][after_func_col])
        vuln_id = str(MSR_df.iloc[i][vuln_id_col])

        _, cache = model.run_with_cache_with_saes([secure_code], saes=[sae])
        index = [f"feature_{i}" for i in range(sae.cfg.d_sae)]
        print(cache.keys())
        feature_activation_df = pd.DataFrame(
            cache["blocks" + "." + str(layer) + "." + CACHE_COMPONENT][0, -1, :]
            .cpu()
            .numpy(),
            index=index,
        )
        feature_activation_df.columns = ["vulnerable"]

        _, cache = model.run_with_cache_with_saes([vulnerable_code], saes=[sae])
        index = [f"feature_{i}" for i in range(sae.cfg.d_sae)]

        feature_activation_df["secure"] = (
            cache["blocks" + "." + str(layer) + "." + CACHE_COMPONENT][0, -1, :]
            .cpu()
            .numpy()
        )

        safe_values = feature_activation_df["secure"].values
        vuln_values = feature_activation_df["vulnerable"].values

        activations = ActivationsSchema(
            vuln_id=vuln_id,
            secure_code=secure_code,
            vulnerable_code=vulnerable_code,
            secure=safe_values.tolist(),
            vulnerable=vuln_values.tolist(),
            layer=layer,
            sae_config=cfg,
        )
        activations.append_to_jsonl(
            f"{output_dir}activations_layer_{layer}_sae_{SAE_ID}_component_{CACHE_COMPONENT}.jsonl"
        )

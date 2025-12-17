import pandas as pd
import torch
from datasets import load_dataset
from sae_lens import SAE, HookedSAETransformer
from tqdm import trange

from sae_java_bug.logger import logger
from sae_java_bug.sparse_autoencoders.schemas import (
    CachedComponent,
    ModelFamily,
    Release,
    SAEConfig,
)

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


MSR_df = load_dataset(hf_path, split="train").to_pandas()
# Filter only 2 samples
MSR_df = MSR_df.sample(n=2, random_state=42)
cfg = SAEConfig(
    model=ModelFamily.GPT2,
    release=Release.GPT2_JB,
    layer_index=3,
    cached_component=CachedComponent.HOOK_SAE_ACTS_POST,
)

print(cfg.sae_id)
MODEL_ARG = cfg.model.value
RELEASE = cfg.release.value
SAE_ID = cfg.sae_id
CACHE_COMPONENT = cfg.cached_component.value
layer = cfg.layer_index


model = HookedSAETransformer.from_pretrained(MODEL_ARG, device=device)
logger.info("Loading Model...")
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release=RELEASE,
    sae_id=SAE_ID,
    device=device,
)
logger.info("Model loaded")

for i in trange(len(MSR_df)):
    import math

    LIMIT = math.inf
    prompt = [str(MSR_df.iloc[i][after_func_col])]
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > LIMIT:
        print("Skiping")
        continue

    prompt = [str(MSR_df.iloc[i][before_func_col])]
    tokens = model.to_tokens(prompt, prepend_bos=True)
    if tokens.shape[1] > LIMIT:
        print("Skiping")
        continue

    _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])
    index = [f"feature_{i}" for i in range(sae.cfg.d_sae)]

    feature_activation_df = pd.DataFrame(
        cache["blocks" + "." + str(layer) + "." + CACHE_COMPONENT][0, -1, :]
        .cpu()
        .numpy(),
        index=index,
    )
    feature_activation_df.columns = ["vulnerable"]

    prompt = [str(MSR_df.iloc[i][after_func_col])]

    _, cache = model.run_with_cache_with_saes(prompt, saes=[sae])
    index = [f"feature_{i}" for i in range(sae.cfg.d_sae)]

    feature_activation_df["secure"] = (
        cache["blocks" + "." + str(layer) + "." + CACHE_COMPONENT][0, -1, :]
        .cpu()
        .numpy()
    )

    safe_values = feature_activation_df["secure"].values
    vuln_values = feature_activation_df["vulnerable"].values

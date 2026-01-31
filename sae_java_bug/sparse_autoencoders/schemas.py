from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# -------------------------------
# High-level model family enums
# -------------------------------


class ModelFamily(str, Enum):
    GPT2 = "gpt2"
    GEMMA = "google/gemma-2-2b"
    GEMMA3 = "google/gemma-3-1b-pt"
    LLAMA = "llama"
    LLAMA_3_1_8B_INST = "meta-llama/Llama-3.1-8B-Instruct"
    LLAMA_3_2_1B_INST = "meta-llama/Llama-3.2-1B-Instruct"
    DEEPSEEK = "meta-llama/Llama-3.1-8B"
    PYTHIA = "pythia-70m-deduped"
    CODE_LLAMA = "codellama/CodeLlama-7b-hf"
    QWEN_CODER = "Qwen/Qwen2.5-7B-Instruct"

class Release(str, Enum):
    GPT2_JB = "gpt2-small-res-jb"
    GEMMA_SCOPE = "gemma-scope-2b-pt-res-canonical"
    GEMMA3 = "gemma-3-1b-res-matryoshka-dc"
    LLAMA_3_1_8B_INST = "goodfire-llama-3.1-8b-instruct"
    LLAMA_SCOPE = "llama_scope_lxr_32x"
    DEEPSEEK_BASE = "llama_scope_r1_distill"
    PYTHIA_70M = "pythia-70m-deduped-res-sm"
    LLAMA_3_2_1B_INST_CUSTOM = "N/A"
    CODE_LLAMA_7B_HF_CUSTOM = "N/A"
    QWEN_CODER_7B_SECURE_CODE_STRD = "rufimelo/secure_code_qwen_coder_strd_16384"
    QWEN_CODER_7B_SECURE_CODE_GATED = "rufimelo/secure_code_qwen_coder_gated_16384"
class CachedComponent(str, Enum):
    HOOK_SAE_ACTS_POST = "hook_resid_pre.hook_sae_acts_post"
    HOOK_RESID_SAE_ACTS_POST = "hook_resid_post.hook_sae_acts_post"
    HOOK_RESID_SAE_ACTS_PRE = "hook_resid_post.hook_sae_acts_pre"


# -------------------------------
# Parametric SAE ID generators
# -------------------------------


def gpt2_resid_pre_layers(n=12):
    return [f"blocks.{i}.hook_resid_pre" for i in range(n)]


def gemma_canonical_layers(n=25):
    return [f"layer_{i}/width_16k/canonical" for i in range(n)]


def gemma3_matryoshka_layers(n=25):
    return [f"blocks.{i}.hook_resid_post" for i in range(n)]


def llama_scope_layers(n=32):
    return [f"l{i}r_32x" for i in range(n)]

def llama_3_1_8b_inst_layers(n=20):
    return [f"layer_{i}" for i in range(n)]

def deepseek_distill_layers(n=32):
    return [f"l{i}r_400m_slimpajama_400m_openr1_math" for i in range(n)]


def pythia_70m_layers(n=6):
    return [f"blocks.{i}.hook_resid_post" for i in range(n)]


def kodcode_llama_3_2_1b_layers(n=16):
    return [f"blocks.{i}.hook_resid_post" for i in range(n)]

def kodcode_code_llama_layers(n=16):
    return [f"blocks.{i}.hook_resid_post" for i in range(n)]

def qwen_coder_7b_layers(n=28):
    return [f"blocks.{i}.hook_resid_post" for i in range(n)]



# -------------------------------
# Central registry
# -------------------------------

SAE_REGISTRY = {
    ModelFamily.GPT2: {
        Release.GPT2_JB: gpt2_resid_pre_layers(12),
    },
    ModelFamily.GEMMA: {
        Release.GEMMA_SCOPE: gemma_canonical_layers(26),
    },
    ModelFamily.GEMMA3: {
        Release.GEMMA3: gemma3_matryoshka_layers(25),
    },
    ModelFamily.LLAMA: {
        Release.LLAMA_SCOPE: llama_scope_layers(32),
    },
    ModelFamily.DEEPSEEK: {
        Release.DEEPSEEK_BASE: deepseek_distill_layers(32),
    },
    ModelFamily.PYTHIA: {
        Release.PYTHIA_70M: pythia_70m_layers(6),
    },
    ModelFamily.LLAMA_3_1_8B_INST: {
        Release.LLAMA_3_1_8B_INST: llama_3_1_8b_inst_layers(20),
    },
    ModelFamily.LLAMA_3_2_1B_INST: {
        Release.LLAMA_3_2_1B_INST_CUSTOM: kodcode_llama_3_2_1b_layers(16),
    },
    ModelFamily.CODE_LLAMA: {
        Release.CODE_LLAMA_7B_HF_CUSTOM: kodcode_code_llama_layers(16),
    },
    ModelFamily.QWEN_CODER: {
        Release.QWEN_CODER_7B_SECURE_CODE_STRD: qwen_coder_7b_layers(28),
    },
    ModelFamily.QWEN_CODER: {
        Release.QWEN_CODER_7B_SECURE_CODE_GATED: qwen_coder_7b_layers(28),
    },
}


# -------------------------------
# Dataclass config
# -------------------------------


@dataclass
class SAEConfig:
    model: ModelFamily
    release: Release
    cached_component: CachedComponent
    layers_available: list[int]
    local_path_template: Optional[str] = field(default=None)

    @property
    def is_local(self) -> bool:
        return self.local_path_template is not None

    def sae_id(self, layer_index: int) -> str:
        try:
            return SAE_REGISTRY[self.model][self.release][layer_index]
        except (KeyError, IndexError):
            raise ValueError(
                f"Invalid configuration: {self.model=} {self.release=} {layer_index=}"
            )

    def sae_path(self, layer_index: int) -> Path:
        """Returns the local path to load the SAE from disk."""
        if not self.is_local:
            raise ValueError("SAE is not configured for local loading")
        sae_id = self.sae_id(layer_index)
        return Path(self.local_path_template.format(sae_id=sae_id))

    def __str__(self):
        return (
            f"SAEConfig(model={self.model}, "
            f"release={self.release}, "
            f"component={self.cached_component}, "
            f"is_local={self.is_local})"
        )
    
    def model_dump(self) -> dict:
        return {
            "model": self.model.value,
            "release": self.release.value,
            "cached_component": self.cached_component.value,
            "layers_available": self.layers_available,
            "local_path_template": self.local_path_template,
        }


GEMMA3_CONFIG = SAEConfig(
    model=ModelFamily.GEMMA3,
    release=Release.GEMMA3,
    cached_component=CachedComponent.HOOK_RESID_SAE_ACTS_PRE,
    layers_available=[i for i in range(25)],
)


LLAMA_3_1_8B_INST_CONFIG = SAEConfig(
    model=ModelFamily.LLAMA_3_1_8B_INST,
    release=Release.LLAMA_3_1_8B_INST,
    cached_component=CachedComponent.HOOK_RESID_SAE_ACTS_PRE,
    layers_available=[19], # Only layer 19 is available.
)


KODCODE_LLAMA_3_2_1B_CONFIG = SAEConfig(
    model=ModelFamily.LLAMA_3_2_1B_INST,
    release=Release.LLAMA_3_2_1B_INST_CUSTOM,
    cached_component=CachedComponent.HOOK_RESID_SAE_ACTS_POST,
    layers_available=[0],  # Currently only layer 0 is available.
    local_path_template="../artifacts/sae_KodCode_llama-3.2-1b-instruct-{sae_id}",
)

KODCODE_CODE_LLAMA_7B_CONFIG = SAEConfig(
    model=ModelFamily.CODE_LLAMA,
    release=Release.CODE_LLAMA_7B_HF_CUSTOM,
    cached_component=CachedComponent.HOOK_RESID_SAE_ACTS_POST,
    layers_available=[0],  # Currently only layer 0 is available.
    local_path_template="../artifacts/sae_KodCode_codellama-7b-hf_tokenized_{sae_id}",
)

QWEN_CODER_7B_SECURE_CODE_STRD_CONFIG = SAEConfig(
    model=ModelFamily.QWEN_CODER,
    release=Release.QWEN_CODER_7B_SECURE_CODE_STRD,
    cached_component=CachedComponent.HOOK_RESID_SAE_ACTS_POST,
    layers_available=[0, 14, 27],
)

QWEN_CODER_7B_SECURE_CODE_GATED_CONFIG = SAEConfig(
    model=ModelFamily.QWEN_CODER,
    release=Release.QWEN_CODER_7B_SECURE_CODE_GATED,
    cached_component=CachedComponent.HOOK_RESID_SAE_ACTS_POST,
    layers_available=[0, 14, 27],
)
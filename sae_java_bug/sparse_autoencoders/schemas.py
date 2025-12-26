from dataclasses import dataclass
from enum import Enum

# -------------------------------
# High-level model family enums
# -------------------------------


class ModelFamily(str, Enum):
    GPT2 = "gpt2"
    GEMMA = "google/gemma-2-2b"
    GEMMA3 = "google/gemma-3-1b-pt"
    LLAMA = "llama"
    LLAMA_3_1_8B_INST = "meta-llama/Llama-3.1-8B-Instruct"
    DEEPSEEK = "meta-llama/Llama-3.1-8B"
    PYTHIA = "pythia-70m-deduped"


class Release(str, Enum):
    GPT2_JB = "gpt2-small-res-jb"
    GEMMA_SCOPE = "gemma-scope-2b-pt-res-canonical"
    GEMMA3 = "gemma-3-1b-res-matryoshka-dc"
    LLAMA_3_1_8B_INST = "goodfire-llama-3.1-8b-instruct"
    LLAMA_SCOPE = "llama_scope_lxr_32x"
    DEEPSEEK_BASE = "llama_scope_r1_distill"
    PYTHIA_70M = "pythia-70m-deduped-res-sm"

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

def llama_3_1_8b_inst_layers(nlayers=[19]):
    return [f"layer_{i}" for i in range(nlayers)]

def deepseek_distill_layers(n=32):
    return [f"l{i}r_400m_slimpajama_400m_openr1_math" for i in range(n)]


def pythia_70m_layers(n=6):
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

    def sae_id(self, layer_index) -> str:
        try:
            return SAE_REGISTRY[self.model][self.release][layer_index]
        except (KeyError, IndexError):
            raise ValueError(
                f"Invalid configuration: {self.model=} {self.release=} {layer_index=}"
            )

    def __str__(self):
        return (
            f"SAEConfig(model={self.model}, "
            f"release={self.release}, "
            f"component={self.cached_component}, "
        )


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
    layers_available=[19],
)
from dataclasses import dataclass
from enum import Enum

# -------------------------------
# High-level model family enums
# -------------------------------


class ModelFamily(str, Enum):
    GPT2 = "gpt2"
    GEMMA = "gemma"
    LLAMA = "llama"


class Release(str, Enum):
    GPT2_JB = "gpt2-small-res-jb"
    GEMMA_SCOPE = "gemma-scope-2b-pt-res-canonical"
    LLAMA_SCOPE = "llama_scope_lxr_32x"


class CachedComponent(str, Enum):
    HOOK_SAE_ACTS_POST = "hook_resid_pre.hook_sae_acts_post"
    HOOK_RESID_SAE_ACTS_POST = "hook_resid_post.hook_sae_acts_post"


# -------------------------------
# Parametric SAE ID generators
# -------------------------------


def gpt2_resid_pre_layers(n=12):
    return [f"blocks.{i}.hook_resid_pre" for i in range(n)]


def gemma_canonical_layers(n=25):
    return [f"layer_{i}/width_16k/canonical" for i in range(n)]


def llama_scope_layers(n=32):
    return [f"l{i}r_32x" for i in range(n)]


# -------------------------------
# Central registry
# -------------------------------

SAE_REGISTRY = {
    ModelFamily.GPT2: {
        Release.GPT2_JB: gpt2_resid_pre_layers(12),
    },
    ModelFamily.GEMMA: {
        Release.GEMMA_SCOPE: gemma_canonical_layers(25),
    },
    ModelFamily.LLAMA: {
        Release.LLAMA_SCOPE: llama_scope_layers(32),
    },
}


# -------------------------------
# Dataclass config
# -------------------------------


@dataclass
class SAEConfig:
    model: ModelFamily
    release: Release
    layer_index: int
    cached_component: CachedComponent

    @property
    def sae_id(self) -> str:
        try:
            return SAE_REGISTRY[self.model][self.release][self.layer_index]
        except (KeyError, IndexError):
            raise ValueError(
                f"Invalid configuration: {self.model=} {self.release=} {self.layer_index=}"
            )

    def __str__(self):
        return (
            f"SAEConfig(model={self.model}, "
            f"release={self.release}, "
            f"layer={self.layer_index}, "
            f"sae_id={self.sae_id}, "
            f"component={self.cached_component})"
        )

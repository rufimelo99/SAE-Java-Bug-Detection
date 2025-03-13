from enum import Enum


class AvailableModels(str, Enum):
    GPT2_SMALL = "gpt2-small"
    LLAMA3_1 = "meta-llama/Llama-3.1-8B"
    GEMMA2_2B =  "google/gemma-2-2b" 
    GEMMA2_2B_IT =  "google/gemma-2-2b-it" 


class PlotType(str, Enum):
    LAYER_WISE = "layer-wise"
    ATTENTION = "attention"
    ACCUMULATED_RESIDUAL = "accumulated-residual"
    SAE_FEATURE_IMPORTANCE = "sae-feature-importance"

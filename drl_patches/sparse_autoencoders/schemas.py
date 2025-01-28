from enum import Enum


class AvailableModels(str, Enum):
    GPT2_SMALL = "gpt2-small"
    LLAMA3_1 = "meta-llama/Llama-3.1-8B"

# LoRA Fine-Tuning Script Walkthrough

This is a script [fine_tune_lora.py](./fine_tune_lora.py) that allows you to fine-tune a Hugging Face model using LoRA on a specified dataset.

## Setup

First, you need to install the required dependencies. Since your environment is managed, it's recommended to use a virtual environment or `conda`.

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

## Running the Script

You can run the script using `python fine_tune_lora.py`.

### Arguments

- `--model_name_or_path`: (Required) The model ID from Hugging Face (e.g., `gpt2`, `meta-llama/Llama-2-7b-hf`).
- `--dataset_name`: (Required) The dataset name from Hugging Face (e.g., `imdb`, `timdettmers/openassistant-guanaco`).
- `--dataset_text_field`: (Optional) The name of the column containing text data (default: `text`).
- `--output_dir`: (Optional) Directory to save results (default: `./lora-output`).
- `--use_4bit`: (Optional) Enable 4-bit quantization (requires GPU).
- `--dataset_text_field`: (Optional) The name of the column containing text data (default: `text`).
- `--target_modules`: (Optional) List of target modules for LoRA (default: `q_proj` `v_proj`). For GPT2 use `c_attn`.
- `--num_train_epochs`: (Optional) Number of epochs (default: 1).

### Example Usage

To fine-tune `gpt2` on the `imdb` dataset (Note: `gpt2` uses `c_attn`):

```bash
python fine_tune_lora.py --model_name_or_path gpt2 --dataset_name imdb --output_dir ./gpt2-lora --max_steps 100 --target_modules c_attn
```

To fine-tune a Llama 2 model with 4-bit quantization:

```bash
python fine_tune_lora.py --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name timdettmers/openassistant-guanaco --use_4bit --output_dir ./llama2-lora
```

## Outputs

The script will save the LoRA adapter weights and the final model configuration to the specified `output_dir`.

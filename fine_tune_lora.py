#!/usr/bin/env python3
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model with LoRA")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset to use (via the datasets library)")
    parser.add_argument("--output_dir", type=str, default="./lora-output", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="The initial learning rate for AdamW.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", type=int, default=-1, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="The maximum sequence length.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--use_4bit", action="store_true", help="Activate 4-bit precision base model loading")
    parser.add_argument("--dataset_text_field", type=str, default="text", help="The name of the text column in the dataset.")
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "v_proj"], help="List of target modules for LoRA")
    
    args = parser.parse_args()

    print(f"Loading model: {args.model_name_or_path}")
    
    # Quantization Config
    bnb_config = None
    if args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare model for k-bit training if using quantization
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA Config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM", # Assuming Causal LM for now
        target_modules=args.target_modules,
    )

    # Load Dataset
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")

    # Training Arguments (SFTConfig)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field=args.dataset_text_field,
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=100,
        fp16=True if torch.cuda.is_available() else False, # Use fp16 on GPU
        push_to_hub=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output_dir}")
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()


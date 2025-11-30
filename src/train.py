# src/train.py
"""Main training script for GRPO on GSM8K dataset."""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from config import (
    SYSTEM_PROMPT,
    MODEL_NAME,
    TRAINING_CONFIG,
    OUTPUT_DIR,
    RUN_NAME,
)
from rewards import (
    extract_hash_answer,
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
)


def main():
    # 1. Prepare Dataset
    print("Loading dataset...")
    data = load_dataset('openai/gsm8k', 'main')['train']
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })

    # 2. Training Configuration
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        **TRAINING_CONFIG
    )

    # 3. Model & Tokenizer
    print(f"Loading model: {MODEL_NAME}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = "left"

    # 4. Initialize Trainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func
        ],
        args=training_args,
        train_dataset=data,
    )

    # 5. Train
    print("Starting training...")
    trainer.train()
    print("Training finished.")
    

if __name__ == "__main__":
    main()
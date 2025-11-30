# src/config.py
"""Hyperparameter configuration for GRPO training."""

# System prompt for the model
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Training hyperparameters
TRAINING_CONFIG = {
    "learning_rate": 5e-6,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "bf16": False,
    "fp16": False,  # Float32 for stability on T4
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 2,
    "num_generations": 4,
    "max_prompt_length": 256,
    "max_completion_length": 512,
    "max_steps": 400,
    "save_steps": 100,
    "max_grad_norm": 0.1,
    "log_on_each_node": False,
    "use_vllm": False,
    "report_to": "none",
}

# Output directories
OUTPUT_DIR = "outputs/Qwen-0.5B-GRPO"
RUN_NAME = "Qwen-0.5B-GRPO-gsm8k"

# Generalized Reward Policy Optimization (GRPO) for Mathematical Reasoning

This repository contains an implementation of Generalized Reward Policy Optimization (GRPO) applied to the `Qwen2.5-0.5B-Instruct` language model. The project demonstrates how to fine-tune small-scale language models for complex mathematical reasoning tasks using the GSM8K dataset.

The core objective is to align the model's output generation to a structured Chain-of-Thought (CoT) format, ensuring both interpretability of the reasoning process and accuracy of the final numerical result.

## Project Overview

Reinforcement Learning from Human Feedback (RLHF) and its variants are standard for aligning LLMs. This project utilizes GRPO, a policy optimization algorithm that eliminates the need for a separate critic model, thereby reducing the computational overhead.

The training pipeline enforces a strict XML-based output structure:
```xml
<reasoning>
[Step-by-step logic and calculation]
</reasoning>
<answer>
[Final integer result]
</answer>
```

## Key Features & Optimizations

*   **Resource Efficiency:** The training pipeline is specifically optimized for consumer-grade or mid-range cloud hardware (e.g., dual NVIDIA T4 GPUs). It utilizes full-precision (Float32) training to mitigate gradient scaling errors common in mixed-precision training on smaller architectures.
*   **Dependency Management:** Addresses specific binary incompatibility issues between `numpy`, `pandas`, and `scipy` often encountered in updated container environments (such as Google Colab or Kaggle).
*   **Custom Reward Modeling:** Implements a composite reward function that evaluates:
    *   **XML Structure:** Validates the presence and order of reasoning and answer tags.
    *   **Format Compliance:** Checks for strict and soft formatting constraints.
    *   **Correctness:** Verifies the final numerical output against the ground truth.

## Repository Structure

```text
.
├── notebooks/
│   └── train_grpo_gsm8k.ipynb    # Main training pipeline
├── src/
│   ├── config.py                 # Hyperparameter configuration
│   └── rewards.py                # Definition of reward functions
├── .gitignore                    # Configuration for ignored files
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Installation

To reproduce this environment, it is critical to install specific versions of the scientific computing stack to avoid `dtype` size mismatches and protobuf errors.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/victormedinaa/qwen-math-rl.git
    cd qwen-math-rl
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

    *Note: The `requirements.txt` enforces `numpy<2.0` and `protobuf==3.20.3` to ensure stability with the `transformers` and `trl` libraries.*

## Training Configuration

The model is trained using the `GRPOTrainer` from the Hugging Face TRL library. Below are the key hyperparameters used for the reference implementation:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Model** | Qwen2.5-0.5B-Instruct | Base model optimized for instruction following. |
| **Learning Rate** | 5e-6 | Conservative rate to prevent catastrophic forgetting. |
| **Batch Size** | 8 (per device) | Optimized for dual T4 GPUs (Effective global batch size: 16). |
| **Max Steps** | 400 | Sufficient for convergence on the reasoning format. |
| **Context Length** | 256 (Prompt) / 512 (Completion) | Extended completion length to allow for detailed CoT. |
| **Precision** | Float32 | Used to prevent `unscale_gradients` errors on T4 architecture. |

## Usage

The primary training logic is contained within the Jupyter Notebook in the `notebooks/` directory.

1.  Ensure your GPU environment is active.
2.  Open `notebooks/train_grpo_gsm8k.ipynb`.
3.  Execute the cells sequentially. The notebook handles the dataset loading, reward function definition, and the training loop.

## Results

Upon completion of the training steps, the model demonstrates a significant improvement in adhering to the requested XML format. The reward functions guide the model to self-correct its generation strategy, prioritizing logical decomposition of the math problems before outputting the final answer.

## Acknowledgments

*   **DeepSeek AI:** For the original proposal of the GRPO algorithm.
*   **Hugging Face:** For the `trl` and `transformers` libraries.
*   **OpenAI:** For the GSM8K dataset.
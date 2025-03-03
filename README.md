# Qwen 0.5B in GRPO

Training a compact model for mathematical reasoning using reinforcement learning

## Description

This repository features an innovative notebook that brings together the **Qwen-0.5B** model and the **GRPO** (Generalized Reward Policy Optimization) technique. The goal? To train a neural network that can tackle school-level math problems. It leverages the **GSM8K** benchmark and uses **vLLM** to speed up and enhance text generation.

Why is this notebook exciting?  
- **Exploring new methods:** It blends reinforcement learning with language model training, refining responses through smart reward functions.  
- **Structured and creative approach:** It uses an XML-based prompt system to generate chain-of-thought reasoning and final answers, neatly breaking down the model’s thinking process.  
- **Cutting-edge tech in action:** From the accelerated text generation of vLLM to modern libraries like `trl` and `datasets` for RL training, this notebook is all about innovation.

## Main Features

- **RL-based training:** Multiple reward functions evaluate both the reasoning process and the generated answers.  
- **GSM8K benchmark:** Provides a robust set of math problems to test the model's reasoning skills.  
- **Efficient resource usage:** vLLM ensures faster and more efficient text generation, making training sessions smoother.  
- **Structured output:** Enforces a specific format with `<reasoning>` and `<answer>` sections to simplify the evaluation and understanding of the model’s thought process.

## Requirements

Before you dive in, make sure you have the following dependencies installed:

- Python 3.8 or higher
- [vLLM](https://github.com/vllm-project/vllm)  
- [trl](https://github.com/lvwerra/trl)
- [datasets](https://huggingface.co/docs/datasets/)
- [transformers](https://huggingface.co/docs/transformers)
- [torch](https://pytorch.org/)

A `requirements.txt` file is provided to make installation a breeze:

```txt
vllm
trl
datasets
transformers
torch
```

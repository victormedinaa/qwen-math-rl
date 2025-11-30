# src/rewards.py
"""Definition of reward functions for GRPO training."""

import re


def extract_xml_answer(text: str) -> str:
    """Extract the answer from XML-formatted response."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_hash_answer(text: str) -> str | None:
    """Extract answer from GSM8K format (after ####)."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """Reward for correct final answer."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def int_reward_func(completions, **kwargs) -> list[float]:
    """Reward for integer answers."""
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for strict XML format compliance."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward for soft XML format compliance."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    """Reward for presence of XML tags."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for text in contents:
        count = 0.0
        if "<reasoning>" in text:
            count += 0.125
        if "</reasoning>" in text:
            count += 0.125
        if "<answer>" in text:
            count += 0.125
        if "</answer>" in text:
            count += 0.125
        rewards.append(count)
    return rewards

import os
import functools
from tqdm import tqdm
from typing import Callable
from pathlib import Path

import torch
import einops

# from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint
from jaxtyping import Int, BFloat16


def load_tensors(model_path: str, concept: str, layer_name: str) -> torch.Tensor:
    model_name = model_path.split("/")[1]

    if model_name not in ["Qwen-1_8B-chat", "gemma-2b-it"]:
        raise ValueError("Invalid model name. Must be one of: Qwen/Qwen-1_8B-chat, ...")

    if concept not in ["harmfulness", "politeness"]:
        raise ValueError(
            "Invalid direction name. Must be one of: harmfulness, politeness."
        )

    if concept == "harmfulness":
        with_concept_direction_name = "harmful"
        without_concept_direction_name = "harmless"
    elif concept == "politeness":
        with_concept_direction_name = "polite"
        without_concept_direction_name = "impolite"

    with_concept_file_path = (
        Path(__file__).parent.parent.resolve()
        / f"directions/{model_name}/{with_concept_direction_name}/{layer_name}.pt"
    )
    without_concept_file_path = (
        Path(__file__).parent.parent.resolve()
        / f"directions/{model_name}/{without_concept_direction_name}/{layer_name}.pt"
    )

    if not os.path.exists(with_concept_file_path):
        raise ValueError(
            f"File not found. Please check the file path.\n{with_concept_file_path} does not exist"
        )

    if not os.path.exists(without_concept_file_path):
        raise ValueError(
            f"File not found. Please check the file path.\n{without_concept_file_path} does not exist"
        )

    with_concept_mean_activation = torch.load(with_concept_file_path, weights_only=True)
    without_concept_mean_activation = torch.load(
        without_concept_file_path, weights_only=True
    )

    return with_concept_mean_activation, without_concept_mean_activation


def direction_ablation_hook(
    activation: BFloat16[torch.Tensor, "... d_act"],  # noqa: F722
    hook: HookPoint,
    direction: BFloat16[torch.Tensor, "d_act"],  # noqa: F821
):
    assert activation.dtype == direction.dtype

    proj = (
        einops.einsum(
            activation, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
        * direction
    )
    return activation - proj


def get_hook_fn(
    concept: str,
    value: int,
    model_name: str,
    layer_name: str = "blocks.14.hook_resid_pre",
) -> Callable:
    with_concept_mean_activation, without_concept_mean_activation = load_tensors(
        model_path=model_name, concept=concept, layer_name=layer_name
    )

    concept_direction = with_concept_mean_activation - without_concept_mean_activation
    concept_direction = concept_direction / (concept_direction.norm())  # * (value / 5)

    hook_fn = functools.partial(direction_ablation_hook, direction=concept_direction)
    return hook_fn


def serialize_history(
    history: list, user_content_template: str, assistant_content_template: str
) -> str:
    """
    Serialize a list of chat messages into a single string prompt.

    Args:
        history: A list of chat messages, where each message is a gr.ChatMessage with (role, content).
        user_content_template: A string template to wrap around user content.
        assistant_content_template: A string template to append to the end of assistant content.

    Returns:
        A single string prompt that represents the entire chat history.
    """
    prompt = ""
    for message in history:
        role, content = message.role, message.content
        if role == "user":  # wrap user content
            prompt += user_content_template.format(content=content)
        elif role == "assistant":  # append assistant reply
            prompt += assistant_content_template.format(content=content)
        else:
            raise NotImplementedError(f"Role {role} not implemented yet.")
    return prompt


def generate_with_hooks(
    model: HookedTransformer,
    tokens: Int[torch.Tensor, "seq_len"],  # noqa: F722, F821
    max_tokens_generated: int = 30,
    fwd_hooks: list | None = None,
) -> str:
    """
    Generate a response given a HookedTransformer model and a prompt.

    Args:
        model: The HookedTransformer model to use for generation.
        tokens: The input prompt tokens.
        max_tokens_generated: The maximum number of tokens to generate.
        fwd_hooks: The forward hooks to apply during generation.

    Returns:
        The generated text as a string.
    """
    if not fwd_hooks:
        fwd_hooks = []

    history_token_len = tokens.shape[0]

    all_tokens = torch.zeros(
        (history_token_len + max_tokens_generated),
        dtype=torch.long,
        device=tokens.device,
    )
    all_tokens[:history_token_len] = tokens

    for i in tqdm(range(max_tokens_generated)):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_tokens[: history_token_len + i])

            # greedy sampling (temperature=0)
            # logits (batch_size = 1, seq_len, vocab_size)
            next_token = logits[:, -1, :].argmax(dim=-1)
            all_tokens[history_token_len + i] = next_token

    response_tokens: list[str] = model.tokenizer.batch_decode(
        all_tokens[history_token_len:], skip_special_tokens=True
    )
    return "".join(response_tokens).strip()

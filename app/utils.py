import os
import functools
from tqdm import tqdm
from typing import Callable

import torch
import einops

# from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from jaxtyping import Int, Float


def load_tensors(direction_name: str, model_name: str, layer: str) -> torch.Tensor:
    if direction_name not in ["harmful", "harmless", "polite"]:
        raise ValueError(
            "Invalid direction name. Must be one of: harmfulness, descriptiveness, politeness."
        )

    if model_name not in ["Qwen/Qwen-1_8B-chat"]:
        raise ValueError("Invalid model name. Must be one of: Qwen/Qwen-1_8B-chat, ...")

    file_path = f"../directions/{direction_name}/{model_name}/layer_{layer}.pt"
    if not os.path.exists(file_path):
        raise ValueError("File not found. Please check the file path.")

    return torch.load(file_path)


# if "Qwen" in MODEL_PATH:
#     model.tokenizer.padding_side = "left"
#     model.tokenizer.pad_token = "<|extra_0|>"

#     with_concept_mean_activation = load_tensors("harmful", "Qwen/Qwen-1_8B-chat", "14")
#     without_concept_mean_activation = load_tensors(
#         "harmless", "Qwen/Qwen-1_8B-chat", "14"
#     )


# def tokenize_instructions_qwen_chat(
#     tokenizer: AutoTokenizer, instructions: list[str]
# ) -> Int[torch.Tensor, "batch_size seq_len"]:  # type: ignore  # noqa: F722
#     prompts = [
#         QWEN_CHAT_TEMPLATE.format(instruction=instruction)
#         for instruction in instructions
#     ]

#     tokens = tokenizer(
#         prompts, padding=True, truncation=False, return_tensors="pt"
#     ).input_ids
#     return tokens


# tokenize_instructions_fn = functools.partial(
#     tokenize_instructions_qwen_chat, tokenizer=model.tokenizer
# )


def direction_ablation_hook(
    activation: Float[torch.Tensor, "... d_act"],  # noqa: F722
    hook: HookPoint,
    direction: Float[torch.Tensor, "d_act"],  # noqa: F821
):
    """
    Appends the user message to the chat history and generates a dummy model response.
    The model response reflects the current concept scale values.
    """
    generations = []

    for i in tqdm(range(0, len(instructions), batch_size)):
        toks = tokenize_instructions_fn(instructions=instructions[i : i + batch_size])
        generation = _generate_with_hooks(
            model,
            toks,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)

    return generations


def get_hook_fn(concept: str, value: int = 3) -> Callable:
    layer = "14"
    if concept == "harmfulness":
        with_concept_mean_activation = load_tensors(
            "harmful", "Qwen/Qwen-1_8B-chat", layer
        )
        without_concept_mean_activation = load_tensors(
            "harmless", "Qwen/Qwen-1_8B-chat", layer
        )

    concept_direction = with_concept_mean_activation - without_concept_mean_activation
    concept_direction = concept_direction / (concept_direction.norm() * (value / 5))

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


def _generate_with_hooks(
    model: HookedTransformer,
    tokens: Int[torch.Tensor, "seq_len"],  # noqa: F722, F821
    max_tokens_generated: int = 30,
    fwd_hooks=[],
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

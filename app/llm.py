import os
import gradio as gr
import functools
from tqdm import tqdm
from typing import Callable

import torch
import einops
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint

from utils import serialize_history, generate_with_hooks, get_hook_fn

MODEL_PATH = "Qwen/Qwen-1_8B-chat"
DEVICE = "cpu"

QWEN_USER_CONTENT_TEMPLATE = """<|im_start|>user{content}<|im_end|>\n"""
QWEN_ASSISTANT_CONTENT_TEMPLATE = """{content}<|im_end|>\n<|im_start|>assistant\n"""

# MODEL_PATH = "google/gemma-2b-it"

GEMMA_USER_CONTENT_TEMPLATE = """<bos><start_of_turn>user{content}<end_of_turn>\n"""
GEMMA_ASSISTANT_CONTENT_TEMPLATE = (
    """{content}\n<start_of_turn>model"""  # ! TODO: check correctness
)

if "gemma" in MODEL_PATH:
    USER_CONTENT_TEMPLATE = GEMMA_USER_CONTENT_TEMPLATE
    ASSISTANT_CONTENT_TEMPLATE = GEMMA_ASSISTANT_CONTENT_TEMPLATE

elif "Qwen" in MODEL_PATH:
    USER_CONTENT_TEMPLATE = QWEN_USER_CONTENT_TEMPLATE
    ASSISTANT_CONTENT_TEMPLATE = QWEN_ASSISTANT_CONTENT_TEMPLATE

model = HookedTransformer.from_pretrained_no_processing(
    MODEL_PATH,
    device=DEVICE,
    dtype=torch.bfloat16,
)
if "Qwen" in MODEL_PATH:
    model.tokenizer.padding_side = "left"
    model.tokenizer.pad_token = "<|extra_0|>"


def generate_response(
    user_message: str,
    history: list,
    # additional inputs
    model_name: str,
    harmfulness: int,
    descriptiveness: int,
    politeness: int,
) -> tuple[list, str]:
    """
    Appends the user message to the chat history and generates a dummy model response.
    The model response reflects the current concept scale values.
    """
    global MODEL_PATH, model

    # load other model
    if model_name != MODEL_PATH:
        raise NotImplementedError("Model loading not implemented yet.")
        MODEL_PATH = model_name
        model = HookedTransformer.from_pretrained_no_processing(
            MODEL_PATH,
            device=DEVICE,
            dtype=torch.float16,
            default_padding_side="left",
            fp16=True,
        )

    # Append user's message
    history.append(gr.ChatMessage(role="user", content=user_message))

    #########
    # HOOKS #
    #########
    harmfulness_hook_fn = get_hook_fn(concept="harmfulness", value=harmfulness)
    # descriptiveness_hook_fn = get_hook_fn(concept="descriptiveness", value=descriptiveness)
    # politeness_hook_fn = get_hook_fn(concept="politeness", value=politeness)

    # intervention_layers = list(range(model.cfg.n_layers))
    # layer = "17"

    intervention_layers = list(range(model.cfg.n_layers))
    # fwd_hooks = [(utils.get_act_name("resid_pre", '14'), harmfulness_hook_fn)]

    fwd_hooks = [
        (utils.get_act_name(act_name, l), harmfulness_hook_fn)
        for l in intervention_layers
        for act_name in ["resid_pre"]  # "resid_mid", "resid_post"
    ]

    ####################
    # TOKENIZE HISTORY #
    ####################
    history_str = serialize_history(
        history, USER_CONTENT_TEMPLATE, ASSISTANT_CONTENT_TEMPLATE
    )
    tokens = model.tokenizer(
        history_str, padding=True, truncation=False, return_tensors="pt"
    ).input_ids.squeeze(
        0
    )  # remove batch size which is equal to 1 here

    ##############
    # GENERATION #
    ##############
    print(f"Generation:\n\t{history}\t{user_message = }\n\t{harmfulness = }")
    response: str = generate_with_hooks(
        model=model,
        tokens=tokens,
        fwd_hooks=fwd_hooks,
        max_tokens_generated=64,
    )

    # Append model's response to the chat history
    history.append(gr.ChatMessage(role="assistant", content=response))

    return history, ""

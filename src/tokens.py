import logging
import os
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, overload

import baukit
import torch
import transformers
from nnsight import LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.utils.tokenization_utils import set_padding_side

from src.models import ModelandTokenizer, determine_device, unwrap_tokenizer
from src.utils.env_utils import DEFAULT_MODELS_DIR
from src.utils.typing import Tokenizer, TokenizerOutput

logger = logging.getLogger(__name__)


def maybe_prefix_bos(tokenizer, prompt: str) -> str:
    """Prefix prompt with EOS token if model has no special start token."""
    tokenizer = unwrap_tokenizer(tokenizer)
    if hasattr(tokenizer, "bos_token"):
        prefix = tokenizer.bos_token
        if not prompt.startswith(prefix):
            prompt = prefix + " " + prompt
    return prompt


def prepare_offset_mapping(string, tokenized, special_tokens):
    """LLaMA3 tokenizer in Huggingface is buggy. This function is a workaround for the bug."""
    """
    <Test>
    
    prompts = ["The Eiffle Tower is located in", "The Space Needle is located in"]
    inp = prepare_input(
        prompts = prompts,
        tokenizer=mt,
        return_offsets_mapping=True,
        device="cuda"
    )

    i=1 # <to be changed>
    for token_id, offset in zip(inp["input_ids"][i], inp["offset_mapping"][i]):
        print(f"`{tokenizer.decode(token_id)}`, {offset=} | `{prompts[i][offset[0]:offset[1]]}`")

    """
    # logger.debug(f"{special_tokens}")
    offset_mapping = []
    end = 0
    # print(tokenized)
    for token in tokenized:
        # print(f"{string[end:].find(token)} | {end=}, {token=}, {string[end:]}")
        next_tok_idx = string[end:].find(token)
        if token in special_tokens and next_tok_idx == -1:
            offset_mapping.append((end, end))
            continue
        assert next_tok_idx != -1, f"{token} not found in {string[end:]}"
        assert next_tok_idx in [
            0,
            1,
        ], f"{token} not found at the beginning of the string"

        start = end
        end = start + string[end:].find(token) + len(token)
        offset_mapping.append((start, end))
    return offset_mapping


def prepare_input(
    prompts: str | list[str],
    tokenizer: ModelandTokenizer | Tokenizer,
    n_gen_per_prompt: int = 1,
    device: torch.device = "cpu",
    add_bos_token: bool = False,
    return_offsets_mapping=False,
    padding: str = "longest",
    padding_side: Optional[Literal["left", "right"]] = None,
    **kwargs,
) -> TokenizerOutput:
    """Prepare input for the model."""
    if isinstance(tokenizer, ModelandTokenizer):
        device = determine_device(
            tokenizer
        )  # if tokenizer type is ModelandTokenizer, get device and ignore the passed device
    # calculate_offsets = return_offsets_mapping and (
    #     isinstance(tokenizer, ModelandTokenizer) and "llama-3" in tokenizer.name.lower()
    # )
    calculate_offsets = False

    tokenizer = unwrap_tokenizer(tokenizer)
    prompts = [prompts] if isinstance(prompts, str) else prompts
    if add_bos_token:
        prompts = [maybe_prefix_bos(tokenizer, p) for p in prompts]
    prompts = [p for p in prompts for _ in range(n_gen_per_prompt)]

    padding_side = padding_side or tokenizer.padding_side

    with set_padding_side(tokenizer, padding_side):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=padding,
            return_offsets_mapping=return_offsets_mapping,
            **kwargs,
        )

    if calculate_offsets:
        offsets = []
        for i in range(len(prompts)):
            tokenized = [tokenizer.decode(t) for t in inputs["input_ids"][i]]
            offsets.append(
                prepare_offset_mapping(
                    string=prompts[i],
                    tokenized=tokenized,
                    special_tokens=tokenizer.all_special_tokens,
                )
            )
        inputs["offset_mapping"] = torch.tensor(offsets)

    inputs = inputs.to(device)
    return inputs


def find_all_single_token_positions(input_ids, token_ids_to_find):
    bools = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in token_ids_to_find:
        bools |= (input_ids == token_id)
    return torch.argwhere(bools)


def find_token_range(
    string: str,
    substring: str,
    tokenizer: Optional[ModelandTokenizer | Tokenizer] = None,
    occurrence: int = 0,
    offset_mapping: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> tuple[int, int]:
    """Find index range of tokenized string containing tokens for substring.

    The kwargs are forwarded to the tokenizer.

    A simple example:

        string = 'The batman is the night.'
        substring = 'batman'
        tokenizer = ...

        # Example tokenization: ['the', 'bat', '##man', 'is', 'the', 'night']
        assert find_token_range(string, substring, tokenizer) == (1, 3)

    Args:
        string: The string.
        substring: The substring to find token range for.
        tokenizer: The tokenizer. If not set, offset_mapping must be.
        occurrence: The occurence of the substring to look for.
            Zero indexed. Defaults to 0, the first occurrence.
        offset_mapping: Precomputed offset mapping. If not set, tokenizer will be run.

    Raises:
        ValueError: If substring is not actually in string or if banned
            kwargs are specified.

    Returns:
        Tuple[int, int]: The start (inclusive) and end (exclusive) token idx.
    """
    if tokenizer is None and offset_mapping is None:
        raise ValueError("must set either tokenizer= or offset_mapping=")
    if "return_offsets_mapping" in kwargs:
        raise ValueError("cannot set return_offsets_mapping")
    if substring not in string:
        raise ValueError(f'"{substring}" not found in "{string}"')

    # logger.debug(f"Found substring in string {string.count(substring)} times")

    if occurrence < 0:
        # If occurrence is negative, count from the right.
        char_start = string.rindex(substring)
        for _ in range(-1 - occurrence):
            try:
                char_start = string.rindex(substring, 0, char_start)
            except ValueError as error:
                raise ValueError(
                    f"could not find {-occurrence} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    else:
        char_start = string.index(substring)
        for _ in range(occurrence):
            try:
                char_start = string.index(substring, char_start + 1)
            except ValueError as error:
                raise ValueError(
                    f"could not find {occurrence + 1} occurrences "
                    f'of "{substring} in "{string}"'
                ) from error
    char_end = char_start + len(substring)

    # logger.debug(
    #     f"char range: [{char_start}, {char_end}] => `{string[char_start:char_end]}`"
    # )

    if offset_mapping is None:
        assert tokenizer is not None
        tokens = prepare_input(
            string, return_offsets_mapping=True, tokenizer=tokenizer, **kwargs
        )
        offset_mapping = tokens.offset_mapping[0]

    token_start, token_end = None, None
    for index, (token_char_start, token_char_end) in enumerate(offset_mapping):
        # logger.debug(f"{index=} | token range: [{token_char_start}, {token_char_end}]")
        if token_char_start == token_char_end:
            # Skip special tokens # ! Is this the proper way of doing this?
            continue
        if token_start is None:
            if token_char_start <= char_start and token_char_end >= char_start:
                token_start = index
        if token_end is None:
            if token_char_start <= char_end and token_char_end >= char_end:
                token_end = index
                break

    # print(f"{substring=}, {occurrence=} | {token_start=}, {token_end=}")
    assert (
        token_start is not None
    ), "Are you working with Llama-3? Try passing the ModelandTokenizer object as the tokenizer"
    assert token_end is not None
    assert token_start <= token_end
    return (token_start, token_end + 1)


def insert_padding_before_subj(
    inp: TokenizerOutput,
    subj_range: tuple[int, int],
    subj_ends: int,
    pad_id: int,
    fill_attn_mask: bool = False,
):
    """

    Inserts padding tokens before the subject in the query to balance the input tensor.

    TEST:

    for idx, (tok_id, attn_mask) in enumerate(zip(clean_inputs.input_ids[0], clean_inputs.attention_mask[0])):
        print(f"{idx=} [{attn_mask}] | {mt.tokenizer.decode(tok_id)}")

    """
    pad_len = subj_ends - subj_range[1]
    inp["input_ids"] = torch.cat(
        [
            inp.input_ids[:, : subj_range[0]],
            torch.full(
                (1, pad_len),
                pad_id,
                dtype=inp.input_ids.dtype,
                device=inp.input_ids.device,
            ),
            inp.input_ids[:, subj_range[0] :],
        ],
        dim=1,
    )

    inp["attention_mask"] = torch.cat(
        [
            inp.attention_mask[:, : subj_range[0]],
            torch.full(
                (1, pad_len),
                fill_attn_mask,
                dtype=inp.attention_mask.dtype,
                device=inp.attention_mask.device,
            ),
            inp.attention_mask[:, subj_range[0] :],
        ],
        dim=1,
    )
    return inp

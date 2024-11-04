import gc
import logging
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import torch
from dataclasses_json import DataClassJsonMixin
from nnsight import LanguageModel
from tqdm.auto import tqdm

from src.functional import (get_all_module_states, get_module_nnsight,
                            guess_subject, predict_next_token)
from src.models import ModelandTokenizer, is_llama_variant
from src.tokens import (find_token_range, insert_padding_before_subj,
                        prepare_input)
from src.utils.typing import PredictedToken, Tokenizer, TokenizerOutput

logger = logging.getLogger(__name__)


@torch.inference_mode()
def patched_run(
    mt: ModelandTokenizer,
    inputs: TokenizerOutput,
    states: dict[tuple[str, int], torch.Tensor],
    scan: bool = False,
) -> torch.Tensor:
    with mt.trace(inputs, scan=scan, validate=False) as trace:
        for location in states:
            layer_name, token_idx = location
            module = get_module_nnsight(mt, layer_name)
            current_states = (
                module.output if ("mlp" in layer_name) else module.output[0]
            )
            current_states[0, token_idx, :] = states[location]
        logits = mt.output.logits[0][-1].save()
    return logits


def get_window(layer_name_format, idx, window_size, n_layer):
    return [
        layer_name_format.format(i)
        for i in range(
            max(0, idx - window_size // 2), min(n_layer - 1, idx + window_size // 2) + 1
        )
    ]


@torch.inference_mode()
def calculate_indirect_effects(
    mt: ModelandTokenizer,
    locations: list[tuple[int, int]],  # layer_idx, token_idx
    corrupted_input: TokenizerOutput,
    patch_states: dict[
        tuple[str, int], torch.Tensor
    ],  # expects the states to be in clean_states
    patch_ans_t: int,
    layer_name_format: str,
    window_size: int = 1,
) -> dict[tuple[str, int], float]:
    is_first = True
    indirect_effects = {loc: -1 for loc in locations}
    for loc in tqdm(locations):
        layer_names = get_window(layer_name_format, loc[0], window_size, mt.n_layer)
        token_idx = loc[1]
        states = {(l, token_idx): patch_states[(l, token_idx)] for l in layer_names}
        affected_logits = patched_run(
            mt=mt,
            inputs=corrupted_input,
            states=states,
            scan=is_first,
        )
        prob = affected_logits.softmax(dim=-1)[patch_ans_t].item()
        indirect_effects[loc] = prob
        is_first = False
    return indirect_effects


@dataclass
class CausalTracingResult(DataClassJsonMixin):
    patch_input_toks: list[str]
    corrupt_input_toks: list[str]
    trace_start_idx: int
    answer: PredictedToken
    low_score: float
    indirect_effects: torch.Tensor
    normalized: bool
    kind: Literal["residual", "mlp", "attention"] = "residual"
    window: int = 1


@torch.inference_mode()
def trace_important_states(
    mt: ModelandTokenizer,
    prompt_template: str,
    clean_subj: str,
    patched_subj: str,
    clean_input: Optional[TokenizerOutput] = None,
    patched_input: Optional[TokenizerOutput] = None,
    kind: Literal["residual", "mlp", "attention"] = "residual",
    window_size: int = 1,
    normalize=True,
    ignore_few_shot_examples=False,
    few_shot_delimiter="\n",
) -> CausalTracingResult:

    if clean_input is None:
        clean_input = prepare_input(
            prompts=prompt_template.format(clean_subj),
            tokenizer=mt,
            return_offset_mapping=True,
        )
    if patched_input is None:
        patched_input = prepare_input(
            prompts=prompt_template.format(patched_subj),
            tokenizer=mt,
            return_offset_mapping=True,
        )

    clean_subj_range = find_token_range(
        string=prompt_template.format(clean_subj),
        substring=clean_subj,
        tokenizer=mt.tokenizer,
        occurrence=-1,
        offset_mapping=clean_input["offset_mapping"][0],
    )
    patched_subj_range = find_token_range(
        string=prompt_template.format(patched_subj),
        substring=patched_subj,
        tokenizer=mt.tokenizer,
        occurrence=-1,
        offset_mapping=patched_input["offset_mapping"][0],
    )

    if clean_subj_range == patched_subj_range:
        subj_start, subj_end = clean_subj_range
    else:
        subj_end = max(clean_subj_range[1], patched_subj_range[1])
        clean_input = insert_padding_before_subj(
            inp=clean_input,
            subj_range=clean_subj_range,
            subj_ends=subj_end,
            pad_id=mt.tokenizer.pad_token_id,
            fill_attn_mask=True,
        )
        patched_input = insert_padding_before_subj(
            inp=patched_input,
            subj_range=patched_subj_range,
            subj_ends=subj_end,
            pad_id=mt.tokenizer.pad_token_id,
            fill_attn_mask=True,
        )

        clean_subj_shift = subj_end - clean_subj_range[1]
        clean_subj_range = (clean_subj_range[0] + clean_subj_shift, subj_end)
        patched_subj_shift = subj_end - patched_subj_range[1]
        patched_subj_range = (patched_subj_range[0] + patched_subj_shift, subj_end)
        subj_start = min(clean_subj_range[0], patched_subj_range[0])

    assert clean_input.input_ids.size(1) == patched_input.input_ids.size(1)

    trace_start_idx = 0
    if clean_input.input_ids[0][0] == patched_input.input_ids[0][0]:
        ft = clean_input.input_ids[0][0]
        print(ft, mt.tokenizer.all_special_ids)
        if ft in mt.tokenizer.all_special_ids:
            trace_start_idx = 1

    logger.debug(f"{trace_start_idx}")
    if ignore_few_shot_examples:
        prompt = mt.tokenizer.decode(clean_input.input_ids[0][trace_start_idx:])
        print(prompt)
        inp = prepare_input(
            prompts=prompt,
            tokenizer=mt,
            return_offset_mapping=True,
        )
        delim_range = find_token_range(
            string=prompt,
            substring=few_shot_delimiter,
            tokenizer=mt.tokenizer,
            occurrence=-1,
            offset_mapping=inp["offset_mapping"][0],
        )
        if delim_range is not None:
            trace_start_idx = delim_range[1] - 1
            logger.debug(f"trace_start_idx updated to {trace_start_idx}")
        else:
            logger.warn(f"{few_shot_delimiter=} not found in prompt")

    logger.debug(f"{trace_start_idx}")
    # raise NotImplementedError("debugging")

    # base run with the patched subject
    patched_states = get_all_module_states(mt=mt, input=patched_input, kind=kind)
    answer = predict_next_token(mt=mt, inputs=patched_input, k=1)[0][0]
    base_probability = answer.prob
    logger.debug(f"{answer=}")

    # clean run
    clean_answer, track_ans = predict_next_token(
        mt=mt, inputs=clean_input, k=1, token_of_interest=answer.token
    )
    clean_answer = clean_answer[0][0]
    low_probability = track_ans[0][1].prob
    logger.debug(f"{clean_answer=}")
    logger.debug(f"{track_ans=}")

    logger.debug("---------- tracing important states ----------")

    assert (
        answer.token != clean_answer.token
    ), "Answers in the clean and corrupt runs are the same"

    layer_name_format = None
    if kind == "residual":
        layer_name_format = mt.layer_name_format
    elif kind == "mlp":
        layer_name_format = mt.mlp_module_name_format
    elif kind == "attention":
        layer_name_format = mt.attn_module_name_format
    else:
        raise ValueError(f"kind must be one of 'residual', 'mlp', 'attention'")

    # calculate indirect effects in the patched run
    locations = [
        (layer_idx, token_idx)
        for layer_idx in range(mt.n_layer)
        for token_idx in range(trace_start_idx, clean_input.input_ids.size(1))
    ]
    indirect_effects = calculate_indirect_effects(
        mt=mt,
        locations=locations,
        corrupted_input=clean_input,
        patch_states=patched_states,
        patch_ans_t=answer.token_id,
        layer_name_format=layer_name_format,
        window_size=window_size,
    )

    indirect_effect_matrix = []
    for token_idx in range(trace_start_idx, clean_input.input_ids.size(1)):
        indirect_effect_matrix.append(
            [
                indirect_effects[(layer_idx, token_idx)]
                for layer_idx in range(mt.n_layer)
            ]
        )

    indirect_effect_matrix = torch.tensor(indirect_effect_matrix)
    if normalize:
        indirect_effect_matrix = (indirect_effect_matrix - low_probability) / (
            base_probability - low_probability
        )

    return CausalTracingResult(
        patch_input_toks=[
            mt.tokenizer.decode(tok) for tok in patched_input.input_ids[0]
        ],
        corrupt_input_toks=[
            mt.tokenizer.decode(tok) for tok in clean_input.input_ids[0]
        ],
        trace_start_idx=trace_start_idx,
        answer=answer,
        low_score=low_probability,
        indirect_effects=indirect_effect_matrix,
        normalized=normalize,
        kind=kind,
        window=window_size,
    )

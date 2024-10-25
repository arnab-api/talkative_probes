import gc
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional, Union

import torch
import copy

# from anthropic import Anthropic
# from openai import OpenAI
from tqdm import tqdm

from src.models import ModelandTokenizer, is_llama_variant
from src.tokens import find_token_range, prepare_input

# from src.utils.env_utils import CLAUDE_CACHE_DIR, GPT_4O_CACHE_DIR
from src.utils.typing import PredictedToken, Tokenizer, TokenizerOutput, LatentCache

logger = logging.getLogger(__name__)


@torch.inference_mode()
def interpret_logits(
    tokenizer: ModelandTokenizer | Tokenizer,
    logits: torch.Tensor,
    k: int = 5,
) -> list[PredictedToken]:
    tokenizer = unwrap_tokenizer(tokenizer)
    logits = logits.squeeze()
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
    top_k_indices = logits.topk(dim=-1, k=k).indices.squeeze().tolist()

    return [
        PredictedToken(
            token=tokenizer.decode(t),
            prob=probs[t].item(),
            logit=logits[t].item(),
            token_id=t,
        )
        for t in top_k_indices
    ]


@torch.inference_mode()
def logit_lens(
    mt: ModelandTokenizer,
    h: torch.Tensor,
    interested_tokens: tuple[int] = (),
    k: int = 5,
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    with mt.trace(get_dummy_input(mt), scan=True, validate=True) as trace:
        lnf = get_module_nnsight(mt, mt.final_layer_norm_name)
        lnf.input = h.view(1, 1, h.squeeze().shape[0])
        logits = mt.output.logits.save()

    logits = logits.squeeze()
    candidates = interpret_logits(tokenizer=mt, logits=logits, k=k)
    if len(interested_tokens) > 0:
        rank_tokens = logits.argsort(descending=True).tolist()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        interested_logits = {
            t: (
                rank_tokens.index(t) + 1,
                PredictedToken(
                    token=mt.tokenizer.decode(t),
                    prob=probs[t].item(),
                    logit=logits[t].item(),
                    token_id=t,
                ),
            )
            for t in interested_tokens
        }
        return candidates, interested_logits
    free_gpu_cache()
    return candidates


@torch.inference_mode()
def patchscope(
    mt: ModelandTokenizer,
    hs: dict[str, torch.Tensor],
    target_prompt: str,
    placeholder_token: str = "placeholder",
    interested_tokens: tuple[int] = (),
    k: int = 5,
) -> (
    list[PredictedToken]
    | tuple[list[PredictedToken], dict[int, tuple[int, PredictedToken]]]
):
    input = prepare_input(
        tokenizer=mt,
        prompts=target_prompt,
        return_offset_mapping=True,
    )
    patches = []
    for i in range(target_prompt.count(placeholder_token)):
        for layer, h in hs.items():
            placeholder_range = find_token_range(
                string=target_prompt,
                substring=placeholder_token,
                tokenizer=mt.tokenizer,
                occurrence=i,
                offset_mapping=input["offset_mapping"][0],
            )
            placeholder_pos = placeholder_range[1] - 1
            logger.debug(
                f"placeholder position: {placeholder_pos} | token: {mt.tokenizer.decode(input['input_ids'][0, placeholder_pos])}"
            )
            patches.append(
                PatchSpec(
                    location=(layer, placeholder_pos),
                    patch=h,
                )
            )
    input.pop("offset_mapping")

    processed_h = get_hs(
        mt=mt,
        input=input,
        locations=[(mt.layer_names[-1], -1)],
        patches=patches,
        return_dict=False,
    )
    return logit_lens(
        mt=mt,
        h=processed_h,
        interested_tokens=interested_tokens,
        k=k,
    )


def untuple(object: Any):
    if isinstance(object, tuple) or (
        "LanguageModelProxy" in str(type(object)) and len(object) > 1
    ):
        return object[0]
    return object


def unwrap_model(mt: ModelandTokenizer | torch.nn.Module) -> torch.nn.Module:
    if isinstance(mt, ModelandTokenizer):
        return mt.model
    if isinstance(mt, torch.nn.Module):
        return mt
    raise ValueError("mt must be a ModelandTokenizer or a torch.nn.Module")


def unwrap_tokenizer(mt: ModelandTokenizer | Tokenizer) -> Tokenizer:
    if isinstance(mt, ModelandTokenizer):
        return mt.tokenizer
    return mt


# useful for logging
def bytes_to_human_readable(
    size: int, unit: Literal["B", "KB", "MB", "GB"] = "MB"
) -> str:
    denom = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30}[unit]
    return f"{size / denom:.3f} {unit}"


def any_is_nontrivial_prefix(predictions: list[str], target: str) -> bool:
    """Return true if any prediction is (case insensitive) prefix of the target."""
    return any(is_nontrivial_prefix(p, target) for p in predictions)


def is_nontrivial_prefix(prediction: str, target: str) -> bool:
    """Return true if prediction is (case insensitive) prefix of the target."""
    target = target.lower().strip()
    prediction = prediction.lower().strip()
    return len(prediction) > 0 and target.startswith(prediction)


def get_tick_marker(value: bool) -> str:
    """Returns a tick or cross marker depending on the value."""
    return "✓" if value else "✗"


def format_whitespace(s: str) -> str:
    """Format whitespace in a string for printing."""
    return s.replace("\n", "\\n").replace("\t", "\\t")


@torch.inference_mode()
def predict_next_token(
    mt: ModelandTokenizer,
    inputs: Union[str, list[str]] | TokenizerOutput,
    k: int = 5,
    batch_size: int = 8,
    token_of_interest: Optional[Union[Union[str, int], list[Union[str, int]]]] = None,
):
    """Predict the next token(s) given the input."""
    if isinstance(inputs, TokenizerOutput):
        if "offset_mapping" in inputs:
            inputs.pop("offset_mapping")
    else:
        inputs = prepare_input(prompts=inputs, tokenizer=mt.tokenizer)
    if token_of_interest is not None:
        token_of_interest = (
            [token_of_interest]
            if not isinstance(token_of_interest, list)
            else token_of_interest
        )
    if token_of_interest is not None:
        assert len(token_of_interest) == len(inputs["input_ids"])
        track_interesting_tokens = []

    predictions = []
    for i in range(0, len(inputs["input_ids"]), batch_size):
        batch_inputs = {
            k: v[i : i + batch_size] if isinstance(v, list) else v
            for k, v in inputs.items()
        }

        with mt.trace(batch_inputs, scan=i == 0) as tr:
            batch_logits = mt.output.logits.save()

        batch_logits = batch_logits[:, -1, :]
        batch_probs = batch_logits.float().softmax(dim=-1)
        batch_topk = batch_probs.topk(k=k, dim=-1)

        for batch_order, (token_ids, token_probs) in enumerate(
            zip(batch_topk.indices, batch_topk.values)
        ):
            predictions.append(
                [
                    PredictedToken(
                        token=mt.tokenizer.decode(token_ids[j]),
                        prob=token_probs[j].item(),
                        logit=batch_logits[batch_order][token_ids[j]].item(),
                        token_id=token_ids[j].item(),
                    )
                    for j in range(k)
                ]
            )

        if token_of_interest is not None:
            _t_idx = 1 if is_llama_variant(mt) else 0
            for j in range(i, i + batch_inputs["input_ids"].shape[0]):
                tok_id = (
                    mt.tokenizer(token_of_interest[j]).input_ids[_t_idx]
                    if type(token_of_interest[j]) == str
                    else token_of_interest[j]
                )
                probs = batch_probs[j]
                order = probs.topk(dim=-1, k=probs.shape[-1]).indices.squeeze()
                prob_tok = probs[tok_id]
                rank = int(torch.where(order == tok_id)[0].item() + 1)
                track_interesting_tokens.append(
                    (
                        rank,
                        PredictedToken(
                            token=mt.tokenizer.decode(tok_id),
                            prob=prob_tok.item(),
                            logit=batch_logits[j - i][tok_id].item(),
                            token_id=tok_id,
                        ),
                    )
                )
        if token_of_interest is not None:
            return predictions, track_interesting_tokens

        return predictions


def get_module_nnsight(model, layer_name):
    layer = model
    for name in layer_name.split("."):
        layer = layer[int(name)] if name.isdigit() else getattr(layer, name)
    return layer


@dataclass(frozen=False)
class PatchSpec:
    location: tuple[str, int]
    patch: torch.Tensor
    clean: Optional[torch.Tensor] = None


@torch.inference_mode()
def get_hs(
    mt: ModelandTokenizer,
    input: str | TokenizerOutput,
    locations: tuple[str, int] | list[tuple[str, int]],
    patches: Optional[PatchSpec | list[PatchSpec]] = None,
    return_dict: bool = False,
) -> torch.Tensor | dict[tuple[str, int], torch.Tensor]:

    if isinstance(input, TokenizerOutput):
        if "offset_mapping" in input:
            input.pop("offset_mapping")
    else:
        input = prepare_input(prompts=input, tokenizer=mt.tokenizer)

    if isinstance(locations, tuple):
        locations = [locations]
    if patches is not None and isinstance(patches, PatchSpec):
        patches = [patches]

    def is_an_attn_head(module_name) -> bool | tuple[int, int]:
        attn_id = mt.attn_module_name_format.split(".")[-1]
        if attn_id not in module_name:
            return False
        if module_name.endswith(attn_id):
            return False

        head_id = module_name.split(".")[-1]
        layer_id = ".".join(module_name.split(".")[:-1])

        return layer_id, int(head_id)

    layer_names = [layer_name for layer_name, _ in locations]
    layer_names = list(set(layer_names))
    layer_states = {layer_name: torch.empty(0) for layer_name in layer_names}
    with mt.trace(input, scan=True) as tracer:
        if patches is not None:
            for cur_patch in patches:
                module_name, index = cur_patch.location
                if is_an_attn_head(module_name) != False:
                    raise NotImplementedError(
                        "patching not supported yet for attn heads"
                    )
                module = get_module_nnsight(mt, module_name)
                current_state = (
                    module.output.save()
                    if ("mlp" in module_name or module_name == mt.embedder_name)
                    else module.output[0].save()
                )
                current_state[0, index, :] = cur_patch.patch

        for layer_name in layer_names:
            if is_an_attn_head(layer_name) == False:
                module = get_module_nnsight(mt, layer_name)
                layer_states[layer_name] = module.output.save()
            else:
                attn_module_name, head_idx = is_an_attn_head(layer_name)
                o_proj_name = attn_module_name + ".o_proj"
                head_dim = mt.n_embd // mt.model.config.num_attention_heads
                o_proj = get_module_nnsight(mt, o_proj_name)
                layer_states[layer_name] = o_proj.input[0][0][
                    :, :, head_idx * head_dim : (head_idx + 1) * head_dim
                ].save()

    hs = {}

    for layer_name, index in locations:
        # print(layer_name, layer_states[layer_name].shape)
        hs[(layer_name, index)] = untuple(layer_states[layer_name])[
            :, index, :
        ].squeeze()

    # print(f"==========> {len(hs)=}")
    if len(hs) == 1 and not return_dict:
        return list(hs.values())[0]
    return hs


@torch.inference_mode
def get_all_module_states(
    mt: ModelandTokenizer,
    input: str | TokenizerOutput,
    kind: Literal["residual", "mlp", "attention"] = "residual",
) -> dict[tuple[str, int], torch.Tensor]:
    if isinstance(input, TokenizerOutput):
        if "offset_mapping" in input:
            input.pop("offset_mapping")
    else:
        input = prepare_input(prompts=input, tokenizer=mt.tokenizer)

    layer_name_format = None
    if kind == "residual":
        layer_name_format = mt.layer_name_format
    elif kind == "mlp":
        layer_name_format = mt.mlp_module_name_format
    elif kind == "attention":
        layer_name_format = mt.attn_module_name_format
    else:
        raise ValueError(f"kind must be one of 'residual', 'mlp', 'attention'")

    layer_and_index = []
    for layer_idx in range(mt.n_layer):
        for token_idx in range(input.input_ids.shape[1]):
            layer_and_index.append((layer_name_format.format(layer_idx), token_idx))

    return get_hs(mt, input, layer_and_index)


def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


# def ask_gpt4o(
#     prompt: str,
# ) -> str:
#     ##################################################
#     client = OpenAI(
#         api_key=os.getenv("OPENAI_KEY"),
#     )
#     MODEL_NAME = "gpt-4o"
#     ##################################################

#     hash_val = hashlib.md5(prompt.encode()).hexdigest()
#     if f"{hash_val}.json" in os.listdir(GPT_4O_CACHE_DIR):
#         logger.debug(f"found cached gpt4o response for {hash_val} - loading")
#         with open(os.path.join(GPT_4O_CACHE_DIR, f"{hash_val}.json"), "r") as f:
#             json_data = json.load(f)
#             return json_data["response"]

#     response = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt},
#         ],
#         temperature=0,
#         max_tokens=4000,
#     )
#     response = response.choices[0].message.content

#     with open(os.path.join(GPT_4O_CACHE_DIR, f"{hash_val}.json"), "w") as f:
#         json.dump(
#             {
#                 "prompt": prompt,
#                 "response": response,
#                 "model": MODEL_NAME,
#                 "hash": hash_val,
#                 "tempraure": 0,
#             },
#             f,
#         )

#     return response


# def ask_claude(
#     prompt: str,
# ) -> str:
#     ##################################################
#     client = Anthropic(
#         api_key=os.getenv("CLAUDE_KEY"),
#     )
#     MODEL_NAME = "claude-3-5-sonnet-20240620"
#     ##################################################

#     hash_val = hashlib.md5(prompt.encode()).hexdigest()
#     if f"{hash_val}.json" in os.listdir(CLAUDE_CACHE_DIR):
#         logger.debug(f"found cached gpt4o response for {hash_val} - loading")
#         with open(os.path.join(CLAUDE_CACHE_DIR, f"{hash_val}.json"), "r") as f:
#             json_data = json.load(f)
#             return json_data["response"]

#     response = client.messages.create(
#         model="claude-3-5-sonnet-20240620",
#         max_tokens=4000,
#         temperature=0,
#         system="You are a helpful assistant.",
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "text",
#                         "text": prompt,
#                     }
#                 ],
#             }
#         ],
#     )
#     response = response.content[0].text

#     with open(os.path.join(CLAUDE_CACHE_DIR, f"{hash_val}.json"), "w") as f:
#         json.dump(
#             {
#                 "prompt": prompt,
#                 "response": response,
#                 "model": MODEL_NAME,
#                 "hash": hash_val,
#                 "tempraure": 0,
#             },
#             f,
#         )

#     return response


# ASK_MODEL = {"gpt4o": ask_gpt4o, "claude": ask_claude}


def free_gpu_cache():
    before = torch.cuda.memory_allocated()
    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_allocated()
    freed = before - after

    # logger.debug(
    #     f"freed {models.bytes_to_human_readable(freed)} | before={models.bytes_to_human_readable(before)} -> after={models.bytes_to_human_readable(after)}"
    # )


def get_dummy_input(
    tokenizer: ModelandTokenizer | Tokenizer,
):
    dummy_prompt = "The quick brown fox"
    return prepare_input(prompts=dummy_prompt, tokenizer=tokenizer)


@torch.inference_mode()
def get_concept_latents(
    mt: ModelandTokenizer,
    queries: list[tuple[str, str]],
    interested_layers: list[str],
    check_answer: bool = True,
) -> list[LatentCache]:
    last_location = (mt.layer_names[-1], -1)
    all_latents = []
    for ques, ans in tqdm(queries):
        inputs = prepare_input(
            prompts=ques,
            tokenizer=mt,
            return_offset_mapping=True,
        )

        query_end = (
            find_token_range(
                string=ques,
                substring=".",
                tokenizer=mt,
                occurrence=-1,
                offset_mapping=inputs["offset_mapping"][0],
            )[1]
            - 1
        )

        hs = get_hs(
            mt=mt,
            input=inputs,
            locations=[(layer, query_end) for layer in interested_layers]
            + [last_location],
            return_dict=True,
        )

        top_prediction = logit_lens(mt=mt, h=hs[last_location])[0]
        if check_answer:
            # query = ques.split("\n")[-1]
            # logger.debug(f"{query} | {top_prediction.token=} | {ans=}")
            if top_prediction.token.strip().lower() != ans.strip().lower():
                continue

        latents = {layer: hs[(layer, query_end)] for layer in interested_layers}

        all_latents.append(
            LatentCache(
                question=ques,
                question_tokenized=[
                    mt.tokenizer.decode(t) for t in inputs["input_ids"][0]
                ],
                answer=ans,
                prediction=top_prediction,
                query_token_idx=query_end,
                latents=latents,
            )
        )

    logger.debug(f"Collected {len(all_latents)} latents, out of {len(queries)}")

    return all_latents


# useful for saving with jsons
def detensorize(inp: dict[Any, Any] | list[dict[Any, Any]]):
    if isinstance(inp, list):
        return [detensorize(i) for i in inp]
    inp = copy.deepcopy(inp)
    for k in inp:
        if isinstance(inp[k], dict):
            inp[k] = detensorize(inp[k])
        elif isinstance(inp[k], torch.Tensor):
            if len(inp[k].shape) == 0:
                inp[k] = inp[k].item()
            else:
                inp[k] = inp[k].tolist()

    free_gpu_cache()
    return inp

from dataclasses import dataclass
from typing import Any
from tqdm import tqdm

import src.functional as functional
import src.tokens as tokens
from src.utils.typing import PredictedToken
from src.models import ModelandTokenizer

def get_source_hs(mt, input_, layers):
    locations = [(mt.layer_name_format.format(layer), input_.input_ids.shape[1] - 1)
                        for layer in layers]
    result = {}
    hs = functional.get_hs(mt=mt, input=input_, locations=locations, return_dict=True)
    for (layer, _), h in hs.items():
        result[layer] = h
    return result

def get_top_interested_token(result_dict):
    top_predicted = None
    for _, (_, pred_token) in result_dict.items():
        # print(pred_token, pred_token.logit)
        if top_predicted is None or pred_token.logit > top_predicted.logit:
            top_predicted = pred_token
    return top_predicted

@dataclass
class PatchscopeConfig:
    # If there are fewer source layers than target layers, source layers will be tiled
    # into target layers. For example, if there is only 1 source layer, it will be patched
    # into all target layers.
    source_layers: list[int]
    # If target layers is None, they will be the same as source layers
    target_layers: list[int] | None

class PatchscopeRunner():
    def __init__(self,
                 mt: ModelandTokenizer,
                 config: PatchscopeConfig,
                 interested_tokens: list[str]):
        self.mt = mt
        self.config = config
        self.interested_tokens = [mt.tokenizer.encode(t)[-1] for t in interested_tokens]
        # print("Interested tokens:")
        # print([mt.tokenizer.decode(t) for t in self.interested_tokens])

    def run(self, source_prompt: str, target_prompt: str) -> PredictedToken:
        if self.config.target_layers is None:
            self.config.target_layers = self.config.source_layers
        assert len(self.config.source_layers) <= len(self.config.target_layers)

        source_input = tokens.prepare_input(source_prompt, self.mt)

        source_hs = get_source_hs(self.mt, source_input, self.config.source_layers)
        target_hs = {}
        for i, target_layer in enumerate(self.config.target_layers):
            source_layer = self.config.source_layers[i % len(self.config.source_layers)]
            target_hs[self.mt.layer_name_format.format(target_layer)] = source_hs[self.mt.layer_name_format.format(source_layer)]

        _, result_dict = functional.patchscope(
            mt = self.mt,
            hs = target_hs,
            target_prompt = target_prompt,
            interested_tokens = self.interested_tokens,
            k = 5)
        return get_top_interested_token(result_dict)

@dataclass
class EvaluationConfig:
    model_key: str
    dataset: str
    target_prompt: str
    label_to_token: dict[Any, str]
    patchscope_config: PatchscopeConfig

@dataclass
class EvaluationResult:
    evaluation_config: EvaluationConfig
    accuracy: float


class EvaluationRunner():
    def __init__(self, mt: ModelandTokenizer, config: EvaluationConfig):
        self.config = config
        self.patchscope_runner = PatchscopeRunner(mt,
                                                  config.patchscope_config,
                                                  config.label_to_token.values())

    def evaluate(self, examples: list[tuple[str, any]]) -> float:
        num_correct = 0
        num_total = 0
        for example, label in tqdm(examples):
            label_token = self.config.label_to_token[label]
            pred_token = self.patchscope_runner.run(example, self.config.target_prompt)
            if pred_token.token == label_token:
                num_correct += 1
            num_total += 1
        accuracy = num_correct / num_total
        return EvaluationResult(evaluation_config=self.config,
                                accuracy=accuracy)

from dataclasses import dataclass
from typing import Any
from tqdm import tqdm

import src.functional as functional
import src.tokens as tokens
from src.utils.typing import PredictedToken
from src.models import ModelandTokenizer
import proto.patchscope_pb2 as patchscope_pb2

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

class PatchscopeRunner():
    def __init__(self,
                 mt: ModelandTokenizer,
                 config: patchscope_pb2.PatchscopeConfig,
                 interested_tokens: list[str]):
        self.mt = mt
        self.config = config
        self.interested_tokens = [mt.tokenizer.encode(t)[-1] for t in interested_tokens]
        # print("Interested tokens:")
        # print([mt.tokenizer.decode(t) for t in self.interested_tokens])

    def run(self, source_prompt: str, target_prompt: str) -> PredictedToken:
        if len(self.config.target_layers) == 0:
            self.config.target_layers.extend(self.config.source_layers)
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


class EvaluationRunner():
    def __init__(self,
                 mt: ModelandTokenizer,
                 config: patchscope_pb2.EvaluationConfig):
        self.config = config
        self.patchscope_runner = PatchscopeRunner(mt,
                                                  config.patchscope_config,
                                                  dict(config.label_to_token).values())

    def evaluate(self,
                 examples: list[tuple[str, any]]) -> float:
        num_correct = 0
        num_total = 0
        for example, label in tqdm(examples):
            label_token = self.config.label_to_token[label]
            pred_token = self.patchscope_runner.run(example, self.config.target_prompt)
            if pred_token.token == label_token:
                num_correct += 1
            num_total += 1
        accuracy = num_correct / num_total
        return patchscope_pb2.EvaluationResult(
            config=self.config,
            accuracy=accuracy,
            num_correct=num_correct,
            num_evaluated=num_total)

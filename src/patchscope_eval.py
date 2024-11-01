from dataclasses import dataclass
from typing import Any
from tqdm import tqdm
import torch

import src.functional as functional
import src.tokens as tokens
from src.utils.typing import PredictedToken
from src.models import ModelandTokenizer
import proto.patchscope_pb2 as patchscope_pb2
from src.dataset_manager import DatasetManager

def get_source_hs(mt, input_, layers):
    locations = [(mt.layer_name_format.format(layer), -1)
                        for layer in layers]
    result = {}
    hs = functional.get_hs(mt=mt, input=input_, locations=locations, return_dict=True)
    for (layer, _), h in hs.items():
        if h.ndim == 1:
            h = h.unsqueeze(0)
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
                 encoder_mt: ModelandTokenizer,
                 decoder_mt: ModelandTokenizer | None,
                 config: patchscope_pb2.PatchscopeConfig,
                 interested_tokens: list[str]):
        self.encoder_mt = encoder_mt
        self.decoder_mt = encoder_mt if decoder_mt is None else decoder_mt
        self.config = config
        self.interested_tokens = interested_tokens
        self.interested_tokens_tensor = torch.tensor(
            [[encoder_mt.tokenizer.encode(t)[-1] for t in interested_tokens]]).cuda()
        # print("Interested tokens:")
        # print([encoder_mt.tokenizer.decode(t) for t in self.interested_tokens])

    def run(self, source_prompts: list[str], target_prompts: list[str]):
        assert len(source_prompts) == len(target_prompts)
        if len(self.config.target_layers) == 0:
            self.config.target_layers.extend(self.config.source_layers)
        assert len(self.config.source_layers) <= len(self.config.target_layers)

        source_input = tokens.prepare_input(source_prompts, self.encoder_mt)

        source_hs = get_source_hs(self.encoder_mt, source_input, self.config.source_layers)
        target_hs = {}
        for i, target_layer in enumerate(self.config.target_layers):
            source_layer = self.config.source_layers[i % len(self.config.source_layers)]
            target_layer_name = self.decoder_mt.layer_name_format.format(target_layer)
            source_layer_name = self.encoder_mt.layer_name_format.format(source_layer)
            target_hs[target_layer_name] = source_hs[source_layer_name]

        logits = functional.patchscope(
            mt = self.decoder_mt,
            hs = target_hs,
            target_prompts = target_prompts)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        predictions = logits.take_along_dim(self.interested_tokens_tensor, dim=-1).argmax(-1)
        result = []
        for pred in predictions:
            result.append(self.interested_tokens[pred])
        return result


class EvaluationRunner():
    def __init__(self,
                 config: patchscope_pb2.EvaluationConfig,
                 encoder_mt: ModelandTokenizer,
                 decoder_mt: ModelandTokenizer | None = None):
        self.config = config
        self.patchscope_runner = PatchscopeRunner(encoder_mt,
                                                  decoder_mt,
                                                  config.patchscope_config,
                                                  config.interested_tokens)

    def evaluate(self,
                 dataset: DatasetManager,
                 max_examples=None) -> float:
        num_correct = 0
        num_total = 0
        for batch in tqdm(dataset):
            source_prompts = []
            target_prompts = []
            answers = []
            for sample in batch:
                source_prompts.append(sample.context)

                placeholder_string = " ".join(["placeholder"] * 5)
                question = placeholder_string + sample.questions[0][1:]
                if self.config.prompt_format != "":
                    question = self.config.prompt_format.format(question)
                target_prompts.append(question)

                answers.append(sample.answers[0])

            predictions = self.patchscope_runner.run(source_prompts, target_prompts)
            assert len(predictions) == len(answers)
            for pred, answer in zip(predictions, answers):
                if pred.strip() == answer.strip():
                    num_correct += 1
                num_total += 1
            
            if max_examples is not None and num_total >= max_examples:
                break
        accuracy = num_correct / num_total
        return patchscope_pb2.EvaluationResult(
            config=self.config,
            accuracy=accuracy,
            num_correct=num_correct,
            num_evaluated=num_total)

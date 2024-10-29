"""Some useful type aliases relevant to this project."""

import pathlib
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy
import torch
import transformers
import transformers.modeling_outputs
from dataclasses_json import DataClassJsonMixin
from nnsight import LanguageModel


ArrayLike = list | tuple | numpy.ndarray | torch.Tensor
PathLike = str | pathlib.Path
Device = str | torch.device

# Throughout this codebase, we use HuggingFace model implementations.
Model = (
    LanguageModel
    | transformers.GPT2LMHeadModel
    | transformers.GPTJForCausalLM
    | transformers.GPTNeoXForCausalLM
    | transformers.LlamaForCausalLM
    | transformers.Gemma2ForCausalLM
    | transformers.GemmaForCausalLM
    | transformers.Qwen2ForCausalLM
)
Tokenizer = transformers.PreTrainedTokenizerFast
TokenizerOffsetMapping = Sequence[tuple[int, int]]
TokenizerOutput = transformers.tokenization_utils_base.BatchEncoding

ModelInput = transformers.BatchEncoding
ModelOutput = transformers.modeling_outputs.CausalLMOutput
ModelGenerateOutput = transformers.generation.utils.GenerateOutput | torch.LongTensor

Layer = int | Literal["emb"] | Literal["ln_f"]

# All strings are also Sequence[str], so we have to distinguish that we
# mean lists or tuples of strings, or sets of strings, not other strings.
StrSequence = list[str] | tuple[str, ...]


@dataclass(frozen=True)
class PredictedToken(DataClassJsonMixin):
    """A predicted token and its probability."""

    token: str
    prob: float
    logit: Optional[float] = None
    token_id: Optional[int] = None

    def __str__(self) -> str:
        return f'"{self.token}" (p={self.prob:.3f})'


@dataclass(frozen=False)
class LatentCache(DataClassJsonMixin):
    context: str
    latents: dict[str, ArrayLike]
    questions: list[str]
    answers: list[str]

    context_tokenized: list[str]
    query_token_idx: int
    prediction: PredictedToken | None = None


@dataclass(frozen=False)
class LatentCacheCollection(DataClassJsonMixin):
    """A collection of latent caches."""

    latents: list[LatentCache] = field(default_factory=list)

    def detensorize(self):
        for latent in self.latents:
            for key, value in latent.latents.items():
                # print(key, type(value), isinstance(value, torch.Tensor))
                if isinstance(value, torch.Tensor) or isinstance(value, numpy.ndarray):
                    latent.latents[key] = value.tolist()

    def retensorize(self, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for latent in self.latents:
            for key, value in latent.latents.items():
                if isinstance(value, list):
                    latent.latents[key] = torch.tensor(value).to(device)

    def __len__(self):
        return len(self.latents)

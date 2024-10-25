from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PatchscopeConfig(_message.Message):
    __slots__ = ("source_layers", "target_layers")
    SOURCE_LAYERS_FIELD_NUMBER: _ClassVar[int]
    TARGET_LAYERS_FIELD_NUMBER: _ClassVar[int]
    source_layers: _containers.RepeatedScalarFieldContainer[int]
    target_layers: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, source_layers: _Optional[_Iterable[int]] = ..., target_layers: _Optional[_Iterable[int]] = ...) -> None: ...

class EvaluationConfig(_message.Message):
    __slots__ = ("model_key", "dataset", "target_prompt", "label_to_token", "patchscope_config")
    class LabelToTokenEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MODEL_KEY_FIELD_NUMBER: _ClassVar[int]
    DATASET_FIELD_NUMBER: _ClassVar[int]
    TARGET_PROMPT_FIELD_NUMBER: _ClassVar[int]
    LABEL_TO_TOKEN_FIELD_NUMBER: _ClassVar[int]
    PATCHSCOPE_CONFIG_FIELD_NUMBER: _ClassVar[int]
    model_key: str
    dataset: str
    target_prompt: str
    label_to_token: _containers.ScalarMap[str, str]
    patchscope_config: PatchscopeConfig
    def __init__(self, model_key: _Optional[str] = ..., dataset: _Optional[str] = ..., target_prompt: _Optional[str] = ..., label_to_token: _Optional[_Mapping[str, str]] = ..., patchscope_config: _Optional[_Union[PatchscopeConfig, _Mapping]] = ...) -> None: ...

class EvaluationResult(_message.Message):
    __slots__ = ("result_set_name", "config", "accuracy", "num_correct", "num_evaluated")
    RESULT_SET_NAME_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    ACCURACY_FIELD_NUMBER: _ClassVar[int]
    NUM_CORRECT_FIELD_NUMBER: _ClassVar[int]
    NUM_EVALUATED_FIELD_NUMBER: _ClassVar[int]
    result_set_name: str
    config: EvaluationConfig
    accuracy: float
    num_correct: int
    num_evaluated: int
    def __init__(self, result_set_name: _Optional[str] = ..., config: _Optional[_Union[EvaluationConfig, _Mapping]] = ..., accuracy: _Optional[float] = ..., num_correct: _Optional[int] = ..., num_evaluated: _Optional[int] = ...) -> None: ...

class EvaluationResults(_message.Message):
    __slots__ = ("results",)
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[EvaluationResult]
    def __init__(self, results: _Optional[_Iterable[_Union[EvaluationResult, _Mapping]]] = ...) -> None: ...

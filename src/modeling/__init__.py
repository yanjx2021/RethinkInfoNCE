import torch
from torch import Tensor
import torch.nn.functional as F

from transformers import AutoConfig, BertConfig

from .bert_dense import BertDense
from .pythia_dense import GPTNeoXDense
from .utils import SIMILARITY_METRICS, POOLING_METHODS

class AutoDenseModel:
    @classmethod
    def from_pretrained(cls, model_path: str, output_dim: int = None, config = None):
        if config is None:
            config = AutoConfig.from_pretrained(model_path)
        if output_dim is not None:
            config.output_dim = output_dim
        else:
            assert hasattr(config, "output_dim"), "Please provide output_dim."
        if config.model_type == "bert":
            config: BertConfig
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0
            model = BertDense.from_pretrained(model_path, config=config)
        elif config.model_type == "gpt_neox":
            model = GPTNeoXDense.from_pretrained(model_path, config=config)
        else:
            raise NotImplementedError()
        return model

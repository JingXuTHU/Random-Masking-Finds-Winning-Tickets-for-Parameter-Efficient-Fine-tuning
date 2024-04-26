import logging
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


def find_module(root_module: nn.Module, key: str):
    """
    Find a module with a specific name in a Transformer model
    From OpenDelta https://github.com/thunlp/OpenDelta
    """
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


class BitfitLinear(torch.nn.Module):
    def __init__(
            self,
            base_Linear: nn.Linear,
            out_features: int
    ):
        super().__init__()
        self.base_Linear = base_Linear
        self.tunable_weight = nn.Parameter(self.base_Linear.bias.new_zeros(out_features))

    def reset_parameters(self):
        pass

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor):
        return F.linear(x, self.base_Linear.weight, self.base_Linear.bias) + self.tunable_weight


class Bitfit:
    def __init__(self, model):
        self.model = model
        if self.model.config.model_type == "opt":
            linear_module_list = ['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc1', 'fc2']
        else:
            raise NotImplementedError

        for key, _ in self.model.named_modules():
            for module_name in linear_module_list:
                if key[-len(module_name):] == module_name:
                    parent_module, sub_key, module = find_module(self.model, key)
                    if module_name == 'fc1':
                        out_features = self.model.config.ffn_dim
                    else:
                        out_features = self.model.config.hidden_size
                    setattr(parent_module, sub_key, BitfitLinear(base_Linear=module, out_features=out_features))

        for n, p in self.model.named_parameters():
            if "tunable_weight" in n:
                p.requires_grad = True
            elif "layer_norm" in n and "bias" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        for n, p in self.model.named_parameters():
            logger.info(f"{n}: {p.requires_grad}")

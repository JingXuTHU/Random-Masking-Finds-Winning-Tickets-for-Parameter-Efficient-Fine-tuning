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


class RandomMaskingLinear(torch.nn.Module):
    def __init__(self, base_Linear: nn.Linear, in_dim: int, out_dim: int, masking_prob: float = 0.0):
        super().__init__()
        self.base_Linear = base_Linear
        self.masking = (torch.rand(out_dim, in_dim) > masking_prob).to(
            base_Linear.weight.device).detach().to(base_Linear.weight.dtype)

        self.tunable_weight = nn.Parameter(self.base_Linear.weight.new_zeros((out_dim, in_dim)))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor):
        tmp1 = F.linear(x, self.base_Linear.weight, self.base_Linear.bias)
        tmp2 = F.linear(x, self.tunable_weight * self.masking)
        return tmp1 + tmp2


class RandomMasking:
    def __init__(self, model, masking_prob):
        self.model = model
        self.masking_prob = masking_prob
        assert 0.0 <= masking_prob <= 1.0

        linear_module_list = ['q_proj', 'v_proj']

        for key, _ in self.model.named_modules():
            for module_name in linear_module_list:
                if key[-len(module_name):] == module_name:
                    parent_module, sub_key, module = find_module(self.model, key)
                    if module_name == 'fc1':
                        out_dim = self.model.config.ffn_dim
                    else:
                        out_dim = self.model.config.hidden_size
                    if module_name == 'fc2':
                        in_dim = self.model.config.ffn_dim
                    else:
                        in_dim = self.model.config.hidden_size

                    setattr(parent_module, sub_key,
                            RandomMaskingLinear(base_Linear=module, in_dim=in_dim, out_dim=out_dim,
                                                masking_prob=self.masking_prob))

        for n, p in model.named_parameters():
            if "tunable" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

        for n, p in model.named_parameters():
            logger.info(f"{n} {p.requires_grad} {p.dtype}")

import logging
import torch
from torch import nn
from torch.nn import functional as F
import math

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


class StructuredMaskingLinear(torch.nn.Module):

    def __init__(
            self,
            base_Linear: nn.Linear,
            in_dim: int,
            out_dim: int,
            masking_prob: float,
            dim_mode: str = 'head',
            **kwargs
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.base_Linear = base_Linear
        self.dim_mode = dim_mode
        self.tunable_length = int(in_dim * out_dim * (1 - masking_prob))
        if self.tunable_length < self.out_dim:
            self.tunable_weight = nn.Parameter(self.base_Linear.weight.new_zeros((self.tunable_length, 1)))
            if self.dim_mode == 'tail':
                self.dim_idx = in_dim - 1
            elif self.dim_mode == 'head':
                self.dim_idx = 0
            elif self.dim_mode == 'random':
                self.dim_idx = torch.randint(0, in_dim, (1,))[0]
        else:
            self.num_cols = math.ceil(in_dim * (1 - masking_prob))
            self.tunable_weight = nn.Parameter(self.base_Linear.weight.new_zeros((out_dim, self.num_cols)))
            self.mask = torch.zeros(out_dim, self.num_cols).to(base_Linear.weight.device).detach().half()
            self.mask[:, :self.num_cols - 1] = 1
            self.mask[:self.tunable_length - self.num_cols * out_dim, -1] = 1

    def reset_parameters(self):
        pass

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor):
        if self.tunable_length < self.out_dim:
            tmp1 = F.linear(x, self.base_Linear.weight, self.base_Linear.bias)
            tmp2 = F.linear(x[:, :, self.dim_idx].view(x.shape[0], x.shape[1], 1), self.tunable_weight)
            tmp1[:, :, :self.tunable_length] += tmp2

            return tmp1
        else:
            tmp1 = F.linear(x, self.base_Linear.weight, self.base_Linear.bias)
            if self.dim_mode == 'head':
                tmp2 = F.linear(x[:, :, :self.num_cols], self.tunable_weight * self.mask)
            elif self.dim_mode == 'tail':
                tmp2 = F.linear(x[:, :, -self.num_cols:], self.tunable_weight * self.mask)
            tmp1 += tmp2
            return tmp1


class StructuredMasking:
    def __init__(self, model, masking_prob, dim_mode='head'):
        self.model = model
        self.masking_prob = masking_prob

        if model.config.model_type == "opt":
            linear_module_list = ['q_proj', 'v_proj']
        else:
            raise NotImplementedError

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
                            StructuredMaskingLinear(base_Linear=module, in_dim=in_dim, out_dim=out_dim,
                                                    masking_prob=masking_prob, dim_mode=dim_mode))

        for n, p in model.named_parameters():
            if "tunable_weight" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

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


class LoRALinear(torch.nn.Module):
    """
    LoRA implemented in a dense layer
    From https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
    """

    def __init__(
            self,
            base_Linear: nn.Linear,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            **kwargs
    ):
        super().__init__()

        self.base_Linear = base_Linear
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.base_Linear.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.base_Linear.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r

            # Freezing the pre-trained weight matrix
            self.base_Linear.weight.requires_grad = False
            if self.base_Linear.bias is not None:
                self.base_Linear.bias.requires_grad = False

            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        if fan_in_fan_out:
            self.base_Linear.weight.data = self.base_Linear.weight.data.transpose(0, 1)

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        result = F.linear(x, T(self.base_Linear.weight), bias=self.base_Linear.bias)
        if self.r > 0:
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0,
                                                                                                  1)) * self.scaling
        return result


class LoRA:

    def __init__(self, model, r, alpha):
        """
        Input:
        r, alpha: LoRA hyperparameters
        """

        self.model = model
        self.hidden_dim = model.config.hidden_size

        if model.config.model_type == "opt":
            attention_name = "attn"
        elif model.config.model_type == "roberta":
            attention_name = "attention"
        elif model.config.model_type == "llama":
            attention_name = "self_attn"
        else:
            raise NotImplementedError

        # Insert LoRA
        for key, _ in model.named_modules():
            if key[-len(attention_name):] == attention_name:
                logger.info(f"Inject lora to: {key}")
                _, _, attn = find_module(model, key)
                device = attn.q_proj.weight.device
                attn.q_proj = LoRALinear(base_Linear=attn.q_proj, in_features=model.config.hidden_size,
                                         out_features=model.config.hidden_size, r=r, lora_alpha=alpha).to(device)
                attn.v_proj = LoRALinear(base_Linear=attn.v_proj, in_features=model.config.hidden_size,
                                         out_features=model.config.hidden_size, r=r, lora_alpha=alpha).to(device)

        # Freeze non-LoRA parameters
        for n, p in model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True

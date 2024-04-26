import logging
import torch
from torch import nn
from torch.nn import functional as F
import re

logger = logging.getLogger(__name__)


def decoder_layer_forward_hook(self, *args, **kwargs):
    hidden_states = args[0]
    if 'attention_mask' in kwargs and kwargs['attention_mask'] is not None:
        attention_mask = kwargs['attention_mask']
        layer_head_mask = kwargs["layer_head_mask"]
        past_key_value = kwargs["past_key_value"]
        output_attentions = kwargs["output_attentions"]
        use_cache = kwargs["use_cache"]
    elif len(args) >= 6:
        attention_mask = args[1]
        layer_head_mask = args[2]
        past_key_value = args[3]
        output_attentions = args[4]
        use_cache = args[5]
    else:
        raise ValueError("decoder_layer_forward_hook: wrong number of arguments")

    device = self.self_attn.out_proj.weight.device
    hidden_states = hidden_states.to(device)

    residual = hidden_states

    # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
    if self.do_layer_norm_before:
        hidden_states = self.self_attn_layer_norm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        past_key_value=past_key_value,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        output_attentions=output_attentions,
    )

    # add the adapter before dropout
    hidden_states = self.adapter1(hidden_states) + hidden_states

    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    # # add the adapter after dropout
    # hidden_states = self.adapter1(hidden_states)+hidden_states

    hidden_states = residual + hidden_states

    # 350m applies layer norm AFTER attention
    if not self.do_layer_norm_before:
        hidden_states = self.self_attn_layer_norm(hidden_states)

    # Fully Connected
    hidden_states_shape = hidden_states.shape
    hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
    residual = hidden_states

    # 125m, 1.7B, ..., 175B applies layer norm BEFORE attention
    if self.do_layer_norm_before:
        hidden_states = self.final_layer_norm(hidden_states)

    hidden_states = self.fc1(hidden_states)
    hidden_states = self.activation_fn(hidden_states)

    hidden_states = self.fc2(hidden_states)

    # add the adapter before dropout
    hidden_states = self.adapter2(hidden_states) + hidden_states

    hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

    # # add the adapter after dropout
    # hidden_states = self.adapter2(hidden_states)+hidden_states

    hidden_states = (residual + hidden_states).view(hidden_states_shape)

    # 350m applies layer norm AFTER attention
    if not self.do_layer_norm_before:
        hidden_states = self.final_layer_norm(hidden_states)

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


class Adapter:

    def __init__(self, model, r, act_type="relu"):
        """
        Input:
        r: size of hidden layer
        """

        self.model = model
        self.hidden_dim = model.config.hidden_size
        self.r = r

        assert (model.config.model_type == "opt")

        for name, module in model.named_modules():
            pattern = r'^model\.decoder\.layers\.\d+$'
            if re.match(pattern, name):
                device = module.self_attn.out_proj.weight.device
                embed_dim = module.embed_dim

                logger.info("inject adapter to {}".format(name))

                if act_type == "None":
                    act_fn = nn.Identity()
                elif act_type == "relu":
                    act_fn = nn.ReLU()
                elif act_type == "gelu":
                    act_fn = nn.GELU()
                elif act_type == "tanh":
                    act_fn = nn.Tanh()

                module.adapter1 = nn.Sequential(
                    nn.Linear(embed_dim, r),
                    act_fn,
                    nn.Linear(r, embed_dim)
                ).to(device).half()

                module.adapter2 = nn.Sequential(
                    nn.Linear(embed_dim, r),
                    act_fn,
                    nn.Linear(r, embed_dim)
                ).to(device).half()

                if act_type == None:
                    nn.init.zeros_(module.adapter1[1].weight)
                    nn.init.zeros_(module.adapter1[1].bias)
                    nn.init.zeros_(module.adapter2[1].weight)
                    nn.init.zeros_(module.adapter2[1].bias)
                else:
                    nn.init.zeros_(module.adapter1[2].weight)
                    nn.init.zeros_(module.adapter1[2].bias)
                    nn.init.zeros_(module.adapter2[2].weight)
                    nn.init.zeros_(module.adapter2[2].bias)

                module.forward = decoder_layer_forward_hook.__get__(module, type(module))

        for name, param in model.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

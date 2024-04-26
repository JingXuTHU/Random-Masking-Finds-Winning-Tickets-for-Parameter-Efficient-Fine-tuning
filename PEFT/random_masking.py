import logging
import torch
from torch import nn
from torch.nn import functional as F
from spops import csr_add, sddmm, csr_transpose

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


def get_sparse_csr(in_dim, out_dim, masking_prob, device):
    # Generate a random matrix and create a mask for non-zero entries
    rand_mat = torch.rand(out_dim, in_dim)
    mask = rand_mat > masking_prob

    # Find indices of non-zero values
    non_zero_indices = torch.nonzero(mask, as_tuple=True)
    row_indices, col_indices = non_zero_indices

    # Extract values using the mask
    values = torch.zeros(row_indices.size(0))

    # Compute row_offsets
    row_counts = torch.bincount(row_indices, minlength=out_dim)
    row_offsets = torch.cat((torch.tensor([0]), torch.cumsum(row_counts, dim=0)))

    # col_idx is already found as col_indices
    col_idx = col_indices

    # Sort rows by the number of non-zeros
    row_idx = torch.argsort(row_counts, descending=True)

    # Conversion to the desired tensors and types
    values = values.to(torch.float16).to(device)  # Ensure values are float
    row_offsets = row_offsets.to(torch.int32).to(device)  # Ensure row_offsets is int32
    col_idx = col_idx.to(torch.int16).to(device)  # Ensure col_idx is int16
    row_idx = row_idx.to(torch.int16).to(device)  # Ensure row_idx is int16, if used

    # Creating a dummy sparse_weight object to mimic returning a similar object as the original function
    sparse_weight = {
        'values': values,
        'row_offsets': row_offsets,
        'col_indices': col_idx,
        'row_indices': row_idx,
    }

    return sparse_weight


class SparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, row_offsets, row_indices, col_indices, dense_weight, bias, input):
        # Save tensors for backward pass
        ctx.save_for_backward(values, row_offsets, row_indices, col_indices, dense_weight, bias, input)

        # Perform the sparse addition
        total_weight = csr_add(values, row_offsets, row_indices, col_indices, dense_weight)

        # Perform the linear operation
        output = F.linear(input, total_weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        values, row_offsets, row_indices, col_indices, dense_weight, bias, input = ctx.saved_tensors
        input_shape = input.shape
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        input = input.reshape(-1, input.shape[-1])

        grad_values = sddmm(row_offsets, row_indices, col_indices, grad_output.T.contiguous(), input.T.contiguous())

        grad_input = (grad_output @ csr_add(values, row_offsets, row_indices, col_indices, dense_weight)).reshape(
            input_shape)

        return grad_values, None, None, None, None, None, grad_input, None


class RandomMaskingLinear(torch.nn.Module):
    def __init__(self, base_Linear: nn.Linear, in_dim: int, out_dim: int, masking_prob: float = 0.0):
        super().__init__()
        self.base_Linear = base_Linear
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Initialization of the sparse matrix components
        sparse_csr = get_sparse_csr(in_dim, out_dim, masking_prob, base_Linear.weight.device)
        self.tunable_weights = nn.Parameter(sparse_csr['values'].to(base_Linear.weight.device))

        # Register row_offsets and col_indices as buffers
        self.register_buffer('row_offsets', sparse_csr['row_offsets'])
        self.register_buffer('col_indices', sparse_csr['col_indices'])
        self.register_buffer('row_indices', sparse_csr['row_indices'])

    def forward(self, input):
        return SparseLinearFunction.apply(self.tunable_weights, self.row_offsets, self.row_indices, self.col_indices,
                                          self.base_Linear.weight, self.base_Linear.bias, input)


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
                                                masking_prob=masking_prob))

        for n, p in model.named_parameters():
            if "tunable_weights" not in n:
                p.requires_grad = False
            else:
                p.requires_grad = True
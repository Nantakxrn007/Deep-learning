import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1d(nn.Module):
    def __init__(self, input_dims, filters, kernel_size, padding='valid'):
        super().__init__()
        self.k_size = kernel_size
        self.pad = padding.lower()
        self.filters = filters
        self.input_dims = input_dims

        # Weight: [filters, input_dims, kernel_size]
        self.w = nn.Parameter(torch.randn(filters, input_dims, kernel_size))
        self.b = nn.Parameter(torch.zeros(filters))

    def forward(self, x):
        # x: [batch_size, seq_len, input_dims] input_dims = n_features but embedding มาแล้ว
        x = x.permute(0, 2, 1)  # -> [batch_size, input_dims, seq_len] because F.conv1d base on [batch_size, seq_len, input_dims]

        if self.pad == 'same':
            pad_total = self.k_size - 1
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x = F.pad(x, (pad_left, pad_right))  # pad along last dim (seq_len)

        # Perform 1D convolution manually using F.conv1d
        out = F.conv1d(x, self.w, self.b) #[batch_size, filters, new_seq_len]
        out = out.permute(0, 2, 1)  # -> [batch_size, new_seq_len, filters]
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class TokenEmbedding(nn.Module):
    """
    TokenEmbedding module:
    Projects input tokens into a higher-dimensional space using a 1D convolution
    followed by a linear layer. Applies layer normalization and tanh activations.
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.d_model = d_model

        # 1D convolution to embed tokens: input channels = c_in, output channels = d_model
        # Kernel size = 2, stride = 2, no bias, using float16 precision, moved to CUDA
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=2,
            bias=False,
            stride=2,
            dtype=torch.float16
        ).to("cuda")

        # Linear layer to project from sequence length 14 to length 2, using float16, on CUDA
        self.linear = nn.Linear(14, 2, dtype=torch.float16).to("cuda")

        # Initialize convolution weights with Kaiming normal initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_in',
                    nonlinearity='leaky_relu'
                )

    def forward(self, x):
        """
        Forward pass:
        1. Prepend the second channel of x to the front along the channel dimension.
        2. Apply layer normalization across all elements of x.
        3. Apply 1D convolution, reshape, and tanh activation.
        4. Apply linear layer, transpose, and final tanh activation.
        """
        # Prepend x[:, 1:2, :] to the beginning of x along channel dimension (dim=1)
        x = torch.cat([x[:, 1:2, :], x], dim=1)

        # Layer normalization: subtract mean and divide by standard deviation
        x = (x - x.mean()) / (torch.sqrt(torch.var(x) + 1e-6))

        # Permute dimensions for convolution: original x shape is (batch, channels, seq_len)
        # After permute: (seq_len, batch, channels) → apply conv → reshape to (batch=1, seq_len_out=14, d_model)
        x = self.tokenConv(x.permute(2, 0, 1)).reshape(1, 14, self.d_model)

        # Apply tanh activation
        x = F.tanh(x)

        # Project with linear layer: input shape (1, 14, d_model) → permute to (1, d_model, 14)
        x = self.linear(x.permute(0, 2, 1)).transpose(1, 2)

        # Final tanh activation
        x = F.tanh(x)
        return x


class ReplicationPad1d(nn.Module):
    """
    ReplicationPad1d module:
    Pads a 1D tensor by replicating border values on both ends.
    """
    def __init__(self, padding) -> None:
        super(ReplicationPad1d, self).__init__()
        # padding is a tuple, e.g., (0, stride) where stride is amount to pad on the right
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass:
        1. Replicate the first element along the last dimension padding[-1] times.
        2. Replicate the last element along the last dimension padding[-1] times.
        3. Concatenate [left_padding, input, right_padding] along the last dimension.
        """
        # Extract left padding: replicate first element padding[-1] times
        replicate_padding1 = input[:, :, 0].unsqueeze(-1).repeat(1, 1, self.padding[-1])

        # Extract right padding: replicate last element padding[-1] times
        replicate_padding2 = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding[-1])

        # Concatenate along the last dimension: (left, input, right)
        output = torch.cat([replicate_padding1, input, replicate_padding2], dim=-1)
        return output


class PatchEmbedding(nn.Module):
    """
    PatchEmbedding module:
    Splits input sequence into overlapping patches via replication padding and
    applies TokenEmbedding to each patch. Finally applies dropout.
    """
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride

        # Replication padding layer: pads input so that sliding window has proper boundaries
        self.padding_patch_layer = ReplicationPad1d((0, stride))

        # TokenEmbedding to embed each patch of length patch_len into d_model-dimensional tokens
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Dropout layer applied after embedding
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass:
        1. Obtain number of variables (channels) from input x.
        2. Compute token embeddings for the input.
        3. Apply dropout and return embeddings along with number of variables.
        """
        # x shape: (batch_size, n_vars, seq_len)
        n_vars = x.shape[1]

        # Apply TokenEmbedding to x
        x = self.value_embedding(x)

        # Apply dropout to embedded tokens
        x = self.dropout(x)

        return x, n_vars
import torch
from torch import nn

import torch
from torch import nn

class AddCoords(nn.Module):
    """
    This is a Pytorch implementation for this model comparison project,
    which is based on the original TF implementation
    of CoordConv by Liu et al. (2018, arXiv:1807.03247) available at
    https://github.com/uber-research/CoordConv/blob/master/CoordConv.py.
    Supports rank-2 tensors only.
    """

    def __init__(self, x_dim=None, y_dim=None, with_r=False):
        """
        Args:
          x_dim: Width of the image.

          y_dim: Height of the image.

          with_r: Whether to add an r-coordinate channel.
        """

        super(AddCoords, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
          input_tensor: (N, C, H, W).
        """

        if len(input_tensor.shape) != 4:
            raise NotImplementedError("Only rank-2 tensors implemented.")

        batch_size_tensor = input_tensor.shape[0]

        # X channel
        xx_ones = torch.ones([1, self.y_dim], dtype=torch.int32)
        xx_ones = xx_ones.unsqueeze(-1)

        xx_range = torch.arange(self.x_dim, dtype=torch.int32).unsqueeze(0)
        xx_range = xx_range.unsqueeze(1)

        xx_channel = torch.matmul(xx_ones, xx_range)
        xx_channel = xx_channel.unsqueeze(-1)

        # xx_channel = torch.arange(self.y_dim, dtype=torch.int32)                # (y_dim)
        # xx_channel = torch.tile(xx_channel, (self.x_dim, batch_size_tensor, 1)) # (x_dim, batch_size_tensor, y_dim)
        # xx_channel = xx_channel.unsqueeze(-1).permute(1, 3, 2, 0)               # (batch_size_tensor, 1, y_dim, x_dim)
        # xx_channel = xx_channel.type_as(input_tensor)                           # Ensure tensor is in the correct device

        # Y-channel
        # yy_channel = torch.arange(self.x_dim, dtype=torch.int32)                # (x_dim)
        # yy_channel = torch.tile(yy_channel, (batch_size_tensor, self.y_dim, 1)) # (batch_size_tensor, y_dim, x_dim)
        # yy_channel = yy_channel.unsqueeze(1)                                    # (batch_size_tensor, 1, y_dim, x_dim)
        # yy_channel = yy_channel.type_as(input_tensor)                           # Ensure tensor is in the correct device
        yy_ones = torch.ones([1, self.x_dim], dtype=torch.int32)
        yy_ones = yy_ones.unsqueeze(1)

        yy_range = torch.arange(self.y_dim, dtype=torch.int32).unsqueeze(0)
        yy_range = yy_range.unsqueeze(-1)

        yy_channel = torch.matmul(yy_range, yy_ones)
        yy_channel = yy_channel.unsqueeze(-1)


        # Normalize to [-1, 1]

        xx_channel = xx_channel.permute(0, 3, 2, 1)
        yy_channel = yy_channel.permute(0, 3, 2, 1)
        xx_channel = xx_channel.type_as(input_tensor)
        yy_channel = yy_channel.type_as(input_tensor)
        
        xx_channel = xx_channel.float() / (self.y_dim - 1)
        yy_channel = yy_channel.float() / (self.x_dim - 1)
        xx_channel = xx_channel*2 - 1
        yy_channel = yy_channel*2 - 1

        xx_channel = xx_channel.repeat(batch_size_tensor, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_tensor, 1, 1, 1)

        ret = torch.cat([input_tensor, xx_channel, yy_channel], dim=1)

        # Extra r-coordinate channel
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                            torch.pow(yy_channel - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)
        ret = ret.type_as(input_tensor)

        return ret

class CoordConv(nn.Module):
    """
    2D-CoordConv layer. Adds (i, j) coordinate information
    to the original input tensor as two additional channels.
    """

    def __init__(self, x_dim, y_dim, with_r, *args, **kwargs):
        """
        Args:
          x_dim: Width of the image.

          y_dim: Height of the image.

          with_r: Whether to add an r-coordinate channel.

          *args, **kwargs: Conv2d parameters.
        """

        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=with_r)
        kwargs['in_channels'] = kwargs['in_channels'] + 2 + with_r
        self.conv = nn.Conv2d(*args, **kwargs)


    def forward(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret

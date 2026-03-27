import torch
import torch.nn as nn
from timm.layers import trunc_normal_, DropPath
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConditionedGRN(nn.Module):
    def __init__(self, dim, g_dim, alpha=0.1, use_norm=True):
        super().__init__()
        self.use_norm = use_norm
        self.alpha = alpha
        self.scale_fc = nn.Linear(g_dim, dim, bias=True)
        self.shift_fc = nn.Linear(g_dim, dim, bias=True)
        nn.init.constant_(self.scale_fc.bias, 1.0)
        nn.init.normal_(self.scale_fc.weight, 0, 1e-3)
        nn.init.constant_(self.shift_fc.bias, 0.0)
        nn.init.normal_(self.shift_fc.weight, 0, 1e-3)
    def forward(self, x, g):
        if self.use_norm:
            Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
            Nx = Gx / (Gx.mean(dim=-1, keepdim=True)+1e-6)
            x_norm = x * Nx
        else:
            x_norm = x
        scale = self.scale_fc(g).unsqueeze(1).unsqueeze(1)
        shift = self.shift_fc(g).unsqueeze(1).unsqueeze(1)
        x_mod = x_norm * scale + shift
        return x + self.alpha * (x_mod - x)

class Block(nn.Module):
    def __init__(self, dim, g_dim=None, drop_path=0.2):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.use_graph = g_dim is not None
        self.grn = ConditionedGRN(4 * dim, g_dim) if self.use_graph else GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x, g=None):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x, g) if self.use_graph else self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x
class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=5, depths=None, dims=None, drop_path_rate=0.2, g_dim=None):
        super().__init__()
        self.depths = depths
        self.g_dim = g_dim
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=3, stride=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            stride = 2 if i == 2 else 1
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=stride),
            )
            self.downsample_layers.append(downsample_layer)
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], g_dim=g_dim, drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
    def forward_features(self, x, g=None):
        for i in range(4):
            x = self.downsample_layers[i](x)
            for blk in self.stages[i]:
                x = blk(x, g) if self.g_dim is not None else blk(x)
        x = self.norm(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        return x
    def forward(self, x, g=None):
        return self.forward_features(x, g)


def convnextv2_atto(g_dim=None):
    return ConvNeXtV2(
        depths=[1,1,3,1],
        dims=[4,8,16,32],
        drop_path_rate=0.2,
        g_dim=g_dim
    )


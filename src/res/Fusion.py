import torch
import torch.nn as nn
import torch.nn.functional as F

class Enhancement_texture_LDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        center = kernel_size // 2
        mask = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
        mask[center, center] = 1.0
        self.register_buffer('center_mask', mask)
        self.base_mask = nn.Parameter(torch.ones_like(self.conv.weight), requires_grad=False)
        self.learnable_mask = nn.Parameter(torch.ones_like(self.conv.weight[:, :, 0, 0]), requires_grad=True)
        self.learnable_theta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(1.0), requires_grad=True)
    def forward(self, x):
        K = self.kernel_size
        normalized_weight = self.conv.weight / (K * K)
        center_mask_weight = self.center_mask * normalized_weight.sum(2).sum(2)[:, :, None, None]
        mask = self.base_mask - self.learnable_theta * self.learnable_mask[:, :, None, None] * center_mask_weight
        out_diff = F.conv2d(
            input=x,
            weight=normalized_weight * mask,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups
        )
        return x + self.beta * out_diff

class Differential_enhance(nn.Module):
    def __init__(self, nf, use_bottleneck=True, dropout=0.2):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Sigmoid()
        if use_bottleneck:
            self.lastconv = nn.Sequential(
                nn.Conv2d(nf, nf // 2, 1),
                nn.GELU(),
                nn.Conv2d(nf // 2, nf, 1)
            )
        else:
            self.lastconv = nn.Conv2d(nf, nf, 1)
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)
    def forward(self, fuse, x1, x2):
        diff = x1 - x2
        weight = self.act(self.global_avgpool(diff))  # [B, C, 1, 1]
        fuse_proj = self.lastconv(fuse)
        diff_enhance = self.dropout(fuse_proj * weight)
        F_1 = x1 + self.beta * diff_enhance
        F_2 = x2 + self.beta * diff_enhance
        return F_1, F_2

class Cross_layer(nn.Module):
    def __init__(self, hidden_dim, g_dim=None, dropout=0.3):
        super().__init__()
        self.use_graph = g_dim is not None
        self.texture_enhance1 = Enhancement_texture_LDC(hidden_dim, hidden_dim)
        self.texture_enhance2 = Enhancement_texture_LDC(hidden_dim, hidden_dim)
        self.Diff_enhance = Differential_enhance(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if self.use_graph:
            self.alpha_mlp = nn.Sequential(
                nn.LayerNorm(g_dim),
                nn.Linear(g_dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:
            self.alpha = nn.Parameter(torch.tensor(0.5))
    def forward(self, Fuse, x1, x2, g_vec=None):
        TX_x1 = self.texture_enhance1(x1)
        TX_x2 = self.texture_enhance2(x2)
        DF_x1, DF_x2 = self.Diff_enhance(Fuse, x1, x2)
        if self.use_graph and g_vec is not None:
            alpha = self.alpha_mlp(g_vec).view(-1, 1, 1, 1)
        else:
            alpha = torch.clamp(self.alpha, 0, 1)
        F_1 = self.dropout(alpha * TX_x1 + (1 - alpha) * DF_x1)
        F_2 = self.dropout(alpha * TX_x2 + (1 - alpha) * DF_x2)
        return F_1 + x1, F_2 + x2


class GDCFM(nn.Module):
    def __init__(self, dim, heads=2, g_dim=None):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.head_dim = dim // heads
        self.use_graph = g_dim is not None
        self.q_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.k_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.v_proj = nn.Conv2d(dim, dim, kernel_size=1)
        mlp_in = dim * 4 + (g_dim if self.use_graph else 0)
        self.out_mlp = nn.Linear(mlp_in, dim)
        self.temperature = nn.Parameter(torch.tensor(self.head_dim ** -0.5))
    def forward(self, feat1, feat2, g_vec=None):
        B, C, H, W = feat1.shape
        def cross_attn(Q, K, V):
            q = Q.view(B, self.heads, self.head_dim, H * W)
            k = K.view(B, self.heads, self.head_dim, H * W)
            v = V.view(B, self.heads, self.head_dim, H * W)
            attn = torch.softmax((q @ k.transpose(-1, -2)) * self.temperature, dim=-1)
            out = (attn @ v).view(B, C, H, W)
            return out
        q1, k2, v2 = self.q_proj(feat1), self.k_proj(feat2), self.v_proj(feat2)
        q2, k1, v1 = self.q_proj(feat2), self.k_proj(feat1), self.v_proj(feat1)
        cross1 = cross_attn(q1, k2, v2)
        cross2 = cross_attn(q2, k1, v1)
        fused1 = feat1 + cross1
        fused2 = feat2 + cross2
        m1, M1 = fused1.mean([2,3]), fused1.amax([2,3])
        m2, M2 = fused2.mean([2,3]), fused2.amax([2,3])
        vec = torch.cat([m1, M1, m2, M2], dim=1)
        if self.use_graph and g_vec is not None:
            vec = torch.cat([vec, g_vec], dim=1)
        return self.out_mlp(vec)



class ResidueDistContactHead(nn.Module):
    def __init__(self, in_ch, out_size=(20, 12), mid_ch=None, dropout=0.5):
        super().__init__()
        self.out_size = out_size
        mid_ch = mid_ch or in_ch
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out_conv = nn.Conv2d(mid_ch, 2, kernel_size=1)
    def forward(self, feat_map):
        x = self.trunk(feat_map)
        out = self.out_conv(x)
        out = F.interpolate(out, size=self.out_size, mode="nearest")
        dist_raw = out[:, 0, :, :]
        contact_logit = out[:, 1, :, :]
        dist_pred = F.softplus(dist_raw)
        return dist_pred, contact_logit

class Fusion_Block(nn.Module):
    def __init__(self, hidden_dim, g_dim=None, out_size=(20, 12), dropout_head=0.5):
        super().__init__()
        self.cross_layer = Cross_layer(hidden_dim, g_dim=g_dim)
        self.cross_fusion = GDCFM(dim=hidden_dim, g_dim=g_dim)
        self.residue_head = ResidueDistContactHead(
            in_ch=hidden_dim,
            out_size=out_size,
            dropout=dropout_head
        )
    def forward(self, input1, input2, g_vec=None):
        if input1.shape[2:] != input2.shape[2:]:
            input2 = F.interpolate(input2, size=input1.shape[2:], mode='nearest')
        F1, F2 = self.cross_layer(input1 + input2, input1, input2, g_vec)
        fused_vec = self.cross_fusion(F1, F2, g_vec)
        dist_pred, contact_logit = self.residue_head(F1)
        return fused_vec, dist_pred, contact_logit


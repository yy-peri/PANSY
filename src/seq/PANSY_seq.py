import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from src.seq.Encoder import convnextv2_atto
from src.seq.Fusion import Fusion_Block


class GlobalGraphEncoder(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=32, out_dim=128, num_edge_types=6, edge_emb_dim=1, dropout=0.5, heads=1, use_in_norm=True,
                 edge_dropout_prob=0.3):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.use_in_norm = use_in_norm
        self.edge_dropout_prob = edge_dropout_prob
        if use_in_norm:
            self.in_norm = nn.LayerNorm(in_dim)
        self.edge_embedding = nn.Embedding(num_edge_types, edge_emb_dim)
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=heads,
                               edge_dim=edge_emb_dim, concat=False, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads,
                               edge_dim=edge_emb_dim, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.pre_out_norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, out_dim)
        self.act = nn.GELU()
    def forward(self, data):
        x = data.x
        if self.use_in_norm:
            x = self.in_norm(x)
        edge_index = data.edge_index
        edge_attr = getattr(data, 'edge_attr', None)
        batch = getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        if edge_attr is not None and edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)
        edge_attr = edge_attr.long()
        edge_feat = self.edge_embedding(edge_attr)  # [E, edge_emb_dim]
        if self.edge_dropout_prob > 0:
            mask = (torch.rand(edge_feat.size(0), 1, device=edge_feat.device) > self.edge_dropout_prob).float()
            edge_feat = edge_feat * mask
        edge_feat = self.dropout(edge_feat)
        x = self.act(self.conv1(x, edge_index, edge_feat))
        x_res = x
        x = self.act(self.conv2(x, edge_index, edge_feat))
        x = x + x_res
        x = global_mean_pool(x, batch)
        x = self.pre_out_norm(x)
        return self.fc_out(x)   # [B, g_dim]


# ConvNeXt
class DualEncoder(nn.Module):
    def __init__(self, encoder_fn, g_dim=None):
        super().__init__()
        self.tcr_pep_encoder = encoder_fn(g_dim=g_dim)
        self.pep_mhc_encoder = encoder_fn(g_dim=g_dim)
    def forward(self, map1, map2, g_vec=None):  # [B, C, H, W], [B, 5, 20, 12], [B, 5, 37, 12], [B, g_dim], [B, 128]
        feat1 = self.tcr_pep_encoder(map1, g_vec)
        feat2 = self.pep_mhc_encoder(map2, g_vec)
        return feat1, feat2  # [B, hidden_dim, H, W]


class TCR_pMHC_binding(nn.Module):
    def __init__(self, g_dim=128, hidden_dim=32, dropout=0.5):
        super().__init__()
        self.graph_encoder = GlobalGraphEncoder(in_dim=6, hidden_dim=32, out_dim=g_dim)
        self.dual_encoder = DualEncoder(convnextv2_atto, g_dim=g_dim)
        self.fusion = Fusion_Block(hidden_dim=hidden_dim, g_dim=g_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )
    def forward(self, map1, map2, graph_data):
        g_vec = self.graph_encoder(graph_data)
        g_vec = g_vec.to(map1.device)
        feat1, feat2 = self.dual_encoder(map1, map2, g_vec)
        fused = self.fusion(feat1, feat2, g_vec)
        logits = self.classifier(fused)
        return logits




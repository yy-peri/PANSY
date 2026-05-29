import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from dataset_seq import get_global_feature, tcr_pep_map, mhc_pep_map, load_mhc_sequences
from torch_geometric.data import Batch


def _pkl_path(pkl_dir, pdb_id: str) -> str:
    return os.path.join(pkl_dir, f"{pdb_id}.pkl")

def _load_pkl(pkl_path: str):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def load_cdr3beta_labels(pkl_dir: str, pdb_id: str, key: str = "cdr3_beta"):
    path = _pkl_path(pkl_dir, pdb_id)
    d = _load_pkl(path)
    if key not in d:
        raise KeyError(f"{path} missing key={key}, keys={list(d.keys())}")
    dist = np.asarray(d[key]["dist"])
    contact = np.asarray(d[key]["contact"])
    return dist, contact


class ResidueDataset(Dataset):
    def __init__(self, df, pkl_dir):
        self.df = df.reset_index(drop=True)
        self.pkl_dir = pkl_dir
        self.mhc_seqs = load_mhc_sequences(self.df["MHC"].tolist())
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdb = row["PDB"]
        cdr3 = row["CDR3"]
        epitope = row["epitope"]
        mhc_seq = self.mhc_seqs[idx]
        tcr_pep_map_feat = tcr_pep_map([cdr3], [epitope]).squeeze(0)   # -> Tensor[...]
        mhc_pep_map_feat = mhc_pep_map([mhc_seq], [epitope]).squeeze(0)
        global_feat = get_global_feature([cdr3], [epitope], [mhc_seq])[0]
        dist, contact = load_cdr3beta_labels(self.pkl_dir, pdb)
        dist = torch.tensor(dist, dtype=torch.float32)
        contact = torch.tensor(contact, dtype=torch.float32)  # BCE 用 float 更自然
        return tcr_pep_map_feat, mhc_pep_map_feat, global_feat, dist, contact


def residue_collate_fn(batch, Lt_m=20, Le_m=12):
    tcr_pep_maps, mhc_pep_maps, graphs, dists, contacts = zip(*batch)
    tcr_pep_maps = torch.stack(tcr_pep_maps, dim=0)
    mhc_pep_maps = torch.stack(mhc_pep_maps, dim=0)
    graph_batch = Batch.from_data_list(list(graphs))
    B = len(dists)
    dist_pad = torch.zeros((B, Lt_m, Le_m), dtype=torch.float32)
    contact_pad = torch.zeros((B, Lt_m, Le_m), dtype=torch.float32)
    mask = torch.zeros((B, Lt_m, Le_m), dtype=torch.bool)
    for i in range(B):
        dist = dists[i]
        contact = contacts[i]
        Lt, Le = dist.shape
        Lt_use = min(Lt, Lt_m)
        Le_use = min(Le, Le_m)
        dist_pad[i, :Lt_use, :Le_use] = dist[:Lt_use, :Le_use]
        contact_pad[i, :Lt_use, :Le_use] = contact[:Lt_use, :Le_use]
        mask[i, :Lt_use, :Le_use] = True
    return tcr_pep_maps, mhc_pep_maps, graph_batch, dist_pad, contact_pad, mask


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g_train = torch.Generator().manual_seed(918)
g_val = torch.Generator().manual_seed(918 + 1)

def build_loaders(df_valid, train_idx, val_idx, pkl_dir, batch_size):
    train_df = df_valid.iloc[train_idx].reset_index(drop=True)
    val_df = df_valid.iloc[val_idx].reset_index(drop=True)

    train_set = ResidueDataset(train_df, pkl_dir)
    val_set = ResidueDataset(val_df, pkl_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=12, collate_fn=residue_collate_fn, worker_init_fn=seed_worker, generator=g_train)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=12, collate_fn=residue_collate_fn, worker_init_fn=seed_worker, generator=g_val)
    return train_loader, val_loader





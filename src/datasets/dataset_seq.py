import numpy as np
import pandas as pd
import peptides
from torch_geometric.data import Data, Batch
import torch
import mhcnames
from torch.utils.data import Dataset

# global
def extract_global_features(seq):
    pep = peptides.Peptide(seq)
    return torch.tensor([
        pep.isoelectric_point(),
        pep.instability_index(),
        pep.aliphatic_index(),
        pep.boman(),
        pep.hydrophobic_moment(),
        pep.molecular_weight(),
    ], dtype=torch.float)

def get_global_feature(tcr_seqs, pep_seqs, mhc_seqs):
    edge_types = {
        (0, 1): 0, (0, 2): 1,
        (1, 0): 2, (1, 2): 3,
        (2, 0): 4, (2, 1): 5
    }
    data_list = []
    for tcr_seq, pep_seq, mhc_seq in zip(tcr_seqs, pep_seqs, mhc_seqs):
        tcr_feat = extract_global_features(tcr_seq)
        pep_feat = extract_global_features(pep_seq)
        mhc_feat = extract_global_features(mhc_seq)
        x = torch.stack([tcr_feat, pep_feat, mhc_feat])  # [3, 6]
        edge_index = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]],
            dtype=torch.long,
            device=x.device
        )
        edge_attr = torch.tensor(
            [[edge_types[(src.item(), dst.item())]] for src, dst in zip(edge_index[0], edge_index[1])],
            dtype=torch.float,
            device=x.device
        )
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_list.append(data)
    return data_list

# local
amino_acid_properties = {
    'A': np.array([1.8, 8.1, 0.0, 0.946, 4.34]),  # Hydrophobicity, Polarity, Charge, Flexibility, Refractivity
    'C': np.array([2.5, 5.5, 0.0, 0.878, 35.77]),
    'D': np.array([-3.5, 13.0, -1.0, 1.089, 12.00]),
    'E': np.array([-3.5, 12.3, -1.0, 1.036, 17.26]),
    'F': np.array([2.8, 5.2, 0.0, 0.912, 29.40]),
    'G': np.array([-0.4, 9.0, 0.0, 1.042, 0.00]),
    'H': np.array([-3.2, 10.4, 0.1, 0.952, 21.81]),
    'I': np.array([4.5, 5.2, 0.0, 0.892, 19.06]),
    'K': np.array([-3.9, 11.3, 1.0, 1.082, 21.29]),
    'L': np.array([3.8, 4.9, 0.0, 0.961, 18.78]),
    'M': np.array([1.9, 5.7, 0.0, 0.862, 21.64]),
    'N': np.array([-3.5, 11.6, 0.0, 1.006, 13.28]),
    'P': np.array([-1.6, 8.0, 0.0, 1.085, 10.93]),
    'Q': np.array([-3.5, 10.5, 0.0, 1.025, 17.56]),
    'R': np.array([-4.5, 10.5, 1.0, 1.028, 26.66]),
    'S': np.array([-0.8, 9.2, 0.0, 1.048, 6.35]),
    'T': np.array([-0.7, 8.6, 0.0, 1.051, 11.01]),
    'V': np.array([4.2, 5.9, 0.0, 0.927, 13.92]),
    'W': np.array([-0.9, 5.4, 0.0, 0.917, 42.53]),
    'Y': np.array([-1.3, 6.2, 0.0, 0.930, 31.53]),
}

def min_max_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)
properties_matrix = np.array(list(amino_acid_properties.values()))
normalized_properties = np.apply_along_axis(min_max_normalization, 0, properties_matrix)
normalized_amino_acid_properties = {
    aa: normalized_properties[i]
    for i, aa in enumerate(amino_acid_properties.keys())
}

queryfile = pd.DataFrame(normalized_amino_acid_properties).T
queryfile.columns = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
amino_acid_features = np.array(list(normalized_amino_acid_properties.values()))  # [20, 5]


# TCR-epitope map
def tcr_pep_map(cdr3s, peptides, length_cdr3=20, length_peptide=12):
    features = list(queryfile.columns)
    aa_to_index = {aa: i for i, aa in enumerate(queryfile.index)}
    feat_matrix = queryfile.values
    B = len(cdr3s)
    num_features = len(features)
    interaction_map = np.zeros((B, num_features, length_cdr3, length_peptide), dtype=np.float32)
    for idx in range(B):
        cdr3 = cdr3s[idx][:length_cdr3].upper()
        pep = peptides[idx][:length_peptide].upper()
        cdr3_feat = np.zeros((length_cdr3, num_features), dtype=np.float32)
        pep_feat = np.zeros((length_peptide, num_features), dtype=np.float32)
        for i, aa in enumerate(cdr3):
            if aa in aa_to_index:
                cdr3_feat[i] = feat_matrix[aa_to_index[aa]]
        for j, aa in enumerate(pep):
            if aa in aa_to_index:
                pep_feat[j] = feat_matrix[aa_to_index[aa]]
        diff = np.abs(cdr3_feat[:, None, :] - pep_feat[None, :, :])  # [L_cdr3, L_pep, F]
        diff = diff.transpose(2, 0, 1)  # → [F, L_cdr3, L_pep]
        interaction_map[idx] = diff
    return torch.tensor(interaction_map)  # [B, num_features, length_cdr3, length_peptide]

# epitope-MHC map
def mhc_pep_map(mhcs, peptides, length_mhc=34, length_peptide=12):
    features = list(queryfile.columns)
    aa_to_index = {aa: i for i, aa in enumerate(queryfile.index)}
    feat_matrix = queryfile.values  # [20, 5]
    B = len(mhcs)
    num_features = len(features)
    interaction_map = np.zeros((B, num_features, length_mhc, length_peptide), dtype=np.float32)
    for idx in range(B):
        mhc = mhcs[idx][:length_mhc].upper()
        pep = peptides[idx][:length_peptide].upper()
        mhc_feat = np.zeros((length_mhc, num_features), dtype=np.float32)
        pep_feat = np.zeros((length_peptide, num_features), dtype=np.float32)
        for i, aa in enumerate(mhc):
            if aa in aa_to_index:
                mhc_feat[i] = feat_matrix[aa_to_index[aa]]
        for j, aa in enumerate(pep):
            if aa in aa_to_index:
                pep_feat[j] = feat_matrix[aa_to_index[aa]]
        diff = np.abs(mhc_feat[:, None, :] - pep_feat[None, :, :])
        diff = diff.transpose(2, 0, 1)  # [F, L_mhc, L_pep]
        interaction_map[idx] = diff
    return torch.tensor(interaction_map)  # [B, num_features, length_mhc, length_peptide]

def load_mhc_sequences(mhc_list, allele_path='data/common_hla_sequence.csv'):
    def preprocess(allele_str):
        allele_str = str(allele_str).strip().split(",")[0]
        allele_str = allele_str.rstrip("LNQS")
        try:
            prefix, rest = allele_str.split("*")
            fields = rest.split(":")
            return f"{prefix}*{fields[0]}:{fields[1]}" if len(fields) >= 2 else allele_str
        except Exception:
            return allele_str

    mhc_trimmed = [preprocess(mhc) for mhc in mhc_list]
    mhc_norm = [mhcnames.normalize_allele_name(mhc) for mhc in mhc_trimmed]
    allele_df = pd.read_csv(allele_path)
    allele_df["allele_trimmed"] = allele_df["allele"].apply(preprocess)
    allele_df["allele_norm"] = allele_df["allele_trimmed"].apply(mhcnames.normalize_allele_name)
    allele_dict = allele_df.set_index("allele_norm")["sequence"]
    mhc_sequences = [allele_dict[mhc] for mhc in mhc_norm]
    return mhc_sequences


class TotalDataset(Dataset):
    def __init__(self, tcr_pep_map, mhc_pep_map, graph_list, labels):
        self.tcr_pep_map = tcr_pep_map
        self.mhc_pep_map = mhc_pep_map
        self.graphs = graph_list
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.tcr_pep_map[idx], self.mhc_pep_map[idx], self.graphs[idx], self.labels[idx]

def collate_fn(batch):
    tcr_pep_map_batch, mhc_pep_map_batch, graph_list, label_batch = zip(*batch)
    tcr_pep_map_batch = torch.stack(tcr_pep_map_batch, dim=0)
    mhc_pep_map_batch = torch.stack(mhc_pep_map_batch, dim=0)
    graph_batch = Batch.from_data_list(list(graph_list))
    label_batch = torch.tensor(label_batch, dtype=torch.long)
    return tcr_pep_map_batch, mhc_pep_map_batch, graph_batch, label_batch

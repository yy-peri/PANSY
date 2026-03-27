import time
import argparse
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
import os
from src.utils import random_seed
from src.res.PANSY_res import TCR_pMHC_binding
from src.datasets.dataset_res import get_global_feature, tcr_pep_map, mhc_pep_map, load_mhc_sequences



class ResidueInferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.mhc_seqs = load_mhc_sequences(self.df["MHC"].tolist())
        if "id" in self.df.columns:
            self.sample_ids = self.df["id"].astype(str).tolist()
        else:
            self.sample_ids = [str(i) for i in range(len(self.df))]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cdr3 = str(row["CDR3"])
        epitope = str(row["epitope"])
        mhc_seq = self.mhc_seqs[idx]
        tcr_pep_map_feat = tcr_pep_map([cdr3], [epitope]).squeeze(0)
        mhc_pep_map_feat = mhc_pep_map([mhc_seq], [epitope]).squeeze(0)
        global_feat = get_global_feature([cdr3], [epitope], [mhc_seq])[0]
        sample_id = self.sample_ids[idx]
        motif = row["motif"]
        meta = {
            "id": sample_id,
            "CDR3": cdr3,
            "epitope": epitope,
            "MHC": str(row["MHC"]),
            "motif": "" if motif is None or (isinstance(motif, float) and np.isnan(motif)) else str(motif),
        }
        return tcr_pep_map_feat, mhc_pep_map_feat, global_feat, meta


def residue_infer_collate_fn(batch, Lt_m=20, Le_m=12):
    tcr_pep_maps, mhc_pep_maps, graphs, metas = zip(*batch)
    tcr_pep_maps = torch.stack(tcr_pep_maps, dim=0)
    mhc_pep_maps = torch.stack(mhc_pep_maps, dim=0)
    graph_batch = Batch.from_data_list(list(graphs))
    B = tcr_pep_maps.shape[0]
    mask = torch.zeros((B, Lt_m, Le_m), dtype=torch.bool)
    for i in range(B):
        cdr3_len = min(len(metas[i]["CDR3"]), Lt_m)
        pep_len = min(len(metas[i]["epitope"]), Le_m)
        mask[i, :cdr3_len, :pep_len] = True
    return tcr_pep_maps, mhc_pep_maps, graph_batch, mask, metas


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# Inference
@torch.no_grad()
def run_inference(model, loader, device, out_npz: str):
    model.eval()
    all_dist_pred = []
    all_contact_prob = []
    all_mask = []
    all_meta = []
    for step, batch in enumerate(loader, 1):
        tcr_pep_maps, mhc_pep_maps, globals_graph, mask, metas = batch
        tcr_pep_maps = tcr_pep_maps.to(device, non_blocking=True)
        mhc_pep_maps = mhc_pep_maps.to(device, non_blocking=True)
        globals_graph = globals_graph.cpu()
        _, dist_pred, contact_logit = model(tcr_pep_maps, mhc_pep_maps, globals_graph)
        dist_pred = dist_pred.detach().cpu()
        contact_prob = torch.sigmoid(contact_logit).detach().cpu()
        all_dist_pred.append(dist_pred)
        all_contact_prob.append(contact_prob)
        all_mask.append(mask.cpu())
        all_meta.extend(list(metas))
        if step % 20 == 0:
            print(f"[INFER] step={step} batch={tcr_pep_maps.shape[0]}")
    dist_pred_full = torch.cat(all_dist_pred, dim=0).numpy()
    contact_prob_full = torch.cat(all_contact_prob, dim=0).numpy()
    mask_full = torch.cat(all_mask, dim=0).numpy()
    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    ids = np.array([m["id"] for m in all_meta], dtype=object)
    cdr3s = np.array([m["CDR3"] for m in all_meta], dtype=object)
    epitopes = np.array([m["epitope"] for m in all_meta], dtype=object)
    mhcs = np.array([m["MHC"] for m in all_meta], dtype=object)
    motifs = np.array([m["motif"] for m in all_meta], dtype=object)
    np.savez_compressed(
        out_npz,
        dist_pred=dist_pred_full,
        contact_prob=contact_prob_full,
        mask=mask_full,
        id=ids,
        CDR3=cdr3s,
        epitope=epitopes,
        MHC=mhcs,
        motif=motifs,
    )
    print(f"[DONE] Saved inference results to: {out_npz}")
    print(f"dist_pred shape: {dist_pred_full.shape}")
    print(f"contact_prob shape: {contact_prob_full.shape}")
    print(f"mask shape: {mask_full.shape}")


def load_model(ckpt_path, device):
    model = TCR_pMHC_binding()
    model.to(device)
    model.graph_encoder.cpu()
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    print(f"[MODEL] Loaded ckpt: {ckpt_path} (strict=True)")
    return model

def main():
    parser = argparse.ArgumentParser("Inference for residue-level contact prediction")
    parser.add_argument("--input_csv", type=str, default='data/prediction.csv',
                        help="CSV with columns: CDR3, epitope, MHC")
    parser.add_argument("--ckpt", type=str, default='checkpoints/PANSY-res.pt',
                        help="Path to a saved model state_dict")
    parser.add_argument("--out_npz", type=str, default='res-outputs/output.npz',
                        help="Output .npz path to save dist_pred/contact_prob/mask/meta.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=918)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    random_seed(args.seed)
    df = pd.read_csv(args.input_csv)
    dataset = ResidueInferenceDataset(df)
    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=residue_infer_collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = load_model(args.ckpt, device=device)
    start = time.time()
    print(f"[START] Inference started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start))}")
    run_inference(model, loader, device=device, out_npz=args.out_npz)
    end = time.time()
    print(f"[END] Inference finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end))}")
    print(f"Elapsed: {end - start:.2f}s")


if __name__ == "__main__":
    main()

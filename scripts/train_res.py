import argparse
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils import EarlyStopping, time_since, random_seed
from PANSY_res import TCR_pMHC_binding
from res_data import build_loaders
from utils import get_scores_dist, get_scores_contact
from sklearn.model_selection import KFold


parser = argparse.ArgumentParser(description='train TCR_pMHC model with residue-level supervision')
parser.add_argument('--input', type=str, default='data/structure.csv')
parser.add_argument('--pkl_dir', type=str, default='data/structure_data')
parser.add_argument('--output_dir', type=str, default='res-outputs/')
parser.add_argument('--model_dir', type=str, default='checkpoints/')
parser.add_argument('--pretrained_ckpt', type=str, default='checkpoints/PANSY-seq.pt', help='path to a pretrained checkpoint (state_dict). If set, load before training.')
parser.add_argument('--num_folds', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--max_epoch', type=int, default=100)
parser.add_argument('--seed', type=int, default=918)
parser.add_argument('--split_mode', type=str, default='random')
parser.add_argument('--finetune_mode', type=str, default='all', choices=['head_only', 'fusion+head', 'all'])
args = parser.parse_args()


# Loss Function
def get_residue_loss(dist_pred, contact_logit, dists, contacts, masks, eps=0.2, w_dist=1.0, w_contact=1.0):
    denom = torch.sum(masks) + 1e-8
    mse_map = F.mse_loss(dist_pred, dists, reduction="none")
    weight = 1.0 / (dists + eps)
    dist_loss = torch.sum(mse_map * masks * weight) / denom
    bce_map = F.binary_cross_entropy_with_logits(contact_logit, contacts, reduction="none")
    contact_loss = torch.sum(bce_map * masks) / denom
    total_loss = w_dist * dist_loss + w_contact * contact_loss
    return {"loss": total_loss, "loss_dist": dist_loss, "loss_contact": contact_loss}


def get_scores(dist_pred, contact_prob, dist, contact, mask):
    avg_dist, metrics_dist = get_scores_dist(dist, dist_pred, mask)
    avg_contact, metrics_contact = get_scores_contact(contact, contact_prob, mask)
    return avg_dist + avg_contact, metrics_dist + metrics_contact


def set_finetune_mode(model, mode):
    # mode: "head_only" | "fusion+head" | "all"
    if mode == "all":
        for p in model.parameters():
            p.requires_grad = True
        return
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fusion.residue_head.parameters():
        p.requires_grad = True
    if mode == "fusion+head":
        for p in model.fusion.cross_layer.parameters():
            p.requires_grad = True


def train_one_epoch(model, train_dataloader, optimizer, device, epoch):
    model.train()
    running = {"loss": 0.0, "loss_dist": 0.0, "loss_contact": 0.0}
    num_steps = 0
    for step, batch in enumerate(train_dataloader, 1):
        tcr_pep_maps, mhc_pep_maps, globals, dists, contacts, masks = batch
        tcr_pep_maps = tcr_pep_maps.to(device, non_blocking=True)
        mhc_pep_maps = mhc_pep_maps.to(device, non_blocking=True)
        globals = globals.cpu()
        dists = dists.to(device, non_blocking=True)
        contacts = contacts.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        _, dist_pred, contact_logit = model(tcr_pep_maps, mhc_pep_maps, globals)
        loss_dict = get_residue_loss(dist_pred, contact_logit, dists, contacts, masks)
        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()
        for k in running:
            running[k] += float(loss_dict[k].detach().cpu())
        num_steps += 1
        if step % 20 == 0:
            print(
                f"[TRAIN] epoch={epoch} step={step} "
                f"loss={running['loss']/num_steps:.4f} "
                f"dist={running['loss_dist']/num_steps:.4f} "
                f"contact={running['loss_contact']/num_steps:.4f}"
            )
    for k in running:
        running[k] /= max(num_steps, 1)
    return running


@torch.no_grad()
def validate_one_epoch(model, val_dataloader, device, epoch):
    model.eval()
    running = {"loss": 0.0, "loss_dist": 0.0, "loss_contact": 0.0}
    num_steps = 0
    all_dist_pred, all_contact_prob = [], []
    all_dist, all_contact, all_mask = [], [], []
    for step, batch in enumerate(val_dataloader, 1):
        tcr_pep_maps, mhc_pep_maps, globals, dists, contacts, masks = batch
        tcr_pep_maps = tcr_pep_maps.to(device, non_blocking=True)
        mhc_pep_maps = mhc_pep_maps.to(device, non_blocking=True)
        globals = globals.cpu()
        dists = dists.to(device, non_blocking=True)
        contacts = contacts.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()
        _, dist_pred, contact_logit = model(tcr_pep_maps, mhc_pep_maps, globals)
        loss_dict = get_residue_loss(dist_pred, contact_logit, dists, contacts, masks)
        for k in running:
            running[k] += float(loss_dict[k].detach().cpu())
        num_steps += 1
        all_dist_pred.append(dist_pred.detach().cpu())
        all_contact_prob.append(torch.sigmoid(contact_logit).detach().cpu())
        all_dist.append(dists.detach().cpu())
        all_contact.append(contacts.detach().cpu())
        all_mask.append(masks.detach().cpu())
        if step % 10 == 0:
            print(
                f"[VAL] epoch={epoch} step={step} "
                f"loss={running['loss']/num_steps:.4f} "
                f"dist={running['loss_dist']/num_steps:.4f} "
                f"contact={running['loss_contact']/num_steps:.4f}"
            )
    for k in running:
        running[k] /= max(num_steps, 1)
    dist_pred_full = torch.cat(all_dist_pred, dim=0)
    contact_prob_full = torch.cat(all_contact_prob, dim=0)
    dist_full = torch.cat(all_dist, dim=0)
    contact_full = torch.cat(all_contact, dim=0)
    mask_full = torch.cat(all_mask, dim=0)
    avg_metrics, metrics_samples = get_scores(dist_pred_full, contact_prob_full, dist_full, contact_full, mask_full)
    print(
        f"[VAL SUMMARY] epoch={epoch} "
        f"loss={running['loss']:.4f} dist={running['loss_dist']:.4f} contact={running['loss_contact']:.4f} | "
        f"pearson={avg_metrics[0]:.4f} mae={avg_metrics[1]:.4f} mape={avg_metrics[2]:.4f} "
        f"mse={avg_metrics[3]:.4f} rmse={avg_metrics[4]:.4f} auc={avg_metrics[5]:.4f} MCC={avg_metrics[6]:.4f}"
    )
    return running, avg_metrics, metrics_samples


@torch.no_grad()
def predict(model, val_loader, device, save_path, val_indices):
    model.eval()
    all_dist_pred, all_contact_prob = [], []
    all_dist, all_contact, all_mask = [], [], []
    for batch in val_loader:
        tcr_pep_maps, mhc_pep_maps, globals, dists, contacts, masks = batch
        tcr_pep_maps = tcr_pep_maps.to(device, non_blocking=True)
        mhc_pep_maps = mhc_pep_maps.to(device, non_blocking=True)
        globals = globals.cpu()
        dists = dists.to(device, non_blocking=True)
        contacts = contacts.to(device, non_blocking=True).float()
        masks = masks.to(device, non_blocking=True).float()
        _, dist_pred, contact_logit = model(tcr_pep_maps, mhc_pep_maps, globals)
        all_dist_pred.append(dist_pred.detach().cpu())
        all_contact_prob.append(torch.sigmoid(contact_logit).detach().cpu())
        all_dist.append(dists.detach().cpu())
        all_contact.append(contacts.detach().cpu())
        all_mask.append(masks.detach().cpu())
    dist_pred_full = torch.cat(all_dist_pred, dim=0).numpy()
    contact_prob_full = torch.cat(all_contact_prob, dim=0).numpy()
    dist_full = torch.cat(all_dist, dim=0).numpy()
    contact_full = torch.cat(all_contact, dim=0).numpy()
    mask_full = torch.cat(all_mask, dim=0).numpy()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        dist_pred=dist_pred_full,
        contact_prob=contact_prob_full,
        dist_gt=dist_full,
        contact_gt=contact_full,
        mask=mask_full,
        sample_idx=np.asarray(val_indices, dtype=np.int64)
    )
    print(f"[PRED] Saved predictions to: {save_path}")


def cv_validation(df_valid, pkl_dir, DEVICE):
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    splitter = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    split_iter = splitter.split(np.arange(len(df_valid)))
    print("[CV] Split mode = random KFold")
    all_fold_results = []
    for fold, (train_idx, val_idx) in enumerate(split_iter, 1):
        print(f"\n--- Fold {fold}/{args.num_folds} ---")
        train_loader, val_loader = build_loaders(
            df_valid=df_valid,
            train_idx=train_idx,
            val_idx=val_idx,
            pkl_dir=pkl_dir,
            batch_size=args.batch_size
        )
        print(f"[Fold {fold}] Train={len(train_idx)}  Val={len(val_idx)}")
        model = TCR_pMHC_binding()
        model.to(DEVICE)
        model.graph_encoder.cpu()
        sd = torch.load(args.pretrained_ckpt, map_location="cpu")
        model.load_state_dict(sd, strict=False)
        print(f"[Fold {fold}] Loaded pretrained ckpt: {args.pretrained_ckpt} (strict=False)")
        set_finetune_mode(model, mode=args.finetune_mode)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )
        best_path = os.path.join(args.model_dir, f"Fold_{fold}_best.pt")
        best_auc_path = os.path.join(args.model_dir, f"Fold_{fold}_best_auc.pt")
        early_stopping = EarlyStopping(
            patience=20,
            verbose=True,
            save_path=best_path,
            mode="min",
            monitor_name="val_loss"
        )
        best_loss = float("inf")
        best_loss_epoch = 0
        best_auc = -np.inf
        best_auc_epoch = 0
        epoch_records = []
        for epoch in range(1, args.max_epoch + 1):
            train_log = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch)
            val_log, avg_metrics, _ = validate_one_epoch(model, val_loader, DEVICE, epoch)
            epoch_records.append({
                "epoch": int(epoch),
                "train_loss": float(train_log["loss"]),
                "train_loss_dist": float(train_log["loss_dist"]),
                "train_loss_contact": float(train_log["loss_contact"]),
                "val_loss": float(val_log["loss"]),
                "val_loss_dist": float(val_log["loss_dist"]),
                "val_loss_contact": float(val_log["loss_contact"]),
                "val_pearson": float(avg_metrics[0]),
                "val_mae": float(avg_metrics[1]),
                "val_mape": float(avg_metrics[2]),
                "val_mse": float(avg_metrics[3]),
                "val_rmse": float(avg_metrics[4]),
                "val_auc": float(avg_metrics[5]),
                "val_mcc": float(avg_metrics[6]),
                "lr": float(optimizer.param_groups[0]["lr"]),
            })
            val_loss = float(val_log["loss"])
            val_auc = float(avg_metrics[5])
            if val_loss < best_loss:
                best_loss = val_loss
                best_loss_epoch = epoch

            if val_auc > best_auc:
                best_auc = val_auc
                best_auc_epoch = epoch
                torch.save(model.state_dict(), best_auc_path)

            lr_scheduler.step(val_loss)
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                break

        records_path = os.path.join(args.output_dir, f"Fold_{fold}_records.csv")
        pd.DataFrame(epoch_records).to_csv(records_path, index=False, encoding="utf-8-sig")

        print(f"[Fold {fold}] Saved epoch records to: {records_path}")
        print(f"[Fold {fold}] Best model saved to: {best_path}")
        print(f"[Fold {fold}] Best auc-model saved to: {best_auc_path}")
        print(f"[Fold {fold}] Best AUC-ROC = {best_auc:.4f} (Epoch {best_auc_epoch})")
        print(f"[Fold {fold}] Best loss = {best_loss:.4f} (Epoch {best_loss_epoch})")

        best_sd = torch.load(best_auc_path, map_location="cpu")
        model.load_state_dict(best_sd, strict=True)
        val_log, avg_metrics, _ = validate_one_epoch(model, val_loader, DEVICE, epoch=0)

        pred_path = os.path.join(args.output_dir, f"predictions/Fold_{fold}_val_predictions.npz")
        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
        predict(model, val_loader, DEVICE, pred_path, val_idx)

        fold_result = {
            "fold": fold,
            "train_size": int(len(train_idx)),
            "val_size": int(len(val_idx)),
            "best_auc": float(best_auc),
            "best_auc_epoch": int(best_auc_epoch),
            "best_loss_epoch": int(best_loss_epoch),
            "val_loss_best": float(val_log["loss"]),
            "val_loss_dist_best": float(val_log["loss_dist"]),
            "val_loss_contact_best": float(val_log["loss_contact"]),
            "val_pearson_best": float(avg_metrics[0]),
            "val_mae_best": float(avg_metrics[1]),
            "val_mape_best": float(avg_metrics[2]),
            "val_mse_best": float(avg_metrics[3]),
            "val_rmse_best": float(avg_metrics[4]),
            "val_auc_best": float(avg_metrics[5]),
            "val_mcc_best": float(avg_metrics[6]),
            "best_path": best_path,
            "best_auc_path": best_auc_path,
            "pred_path": pred_path,
        }
        all_fold_results.append(fold_result)
        print(f"[Fold {fold} DONE] val_loss={fold_result['val_loss_best']:.4f} auc={fold_result['val_auc_best']:.4f}")

    pd.DataFrame(all_fold_results).to_csv(
        os.path.join(args.output_dir, "cv_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pearsons = [x["val_pearson_best"] for x in all_fold_results]
    maes = [x["val_mae_best"] for x in all_fold_results]
    mapes = [x["val_mape_best"] for x in all_fold_results]
    mses = [x["val_mse_best"] for x in all_fold_results]
    rmses = [x["val_rmse_best"] for x in all_fold_results]
    aucs = [x["val_auc_best"] for x in all_fold_results]
    losses = [x["val_loss_best"] for x in all_fold_results]

    print("\n===== Cross-validation Completed =====")
    print(f"[CV SUMMARY] mean val_pearson = {np.mean(pearsons):.4f} ± {np.std(pearsons):.4f}")
    print(f"[CV SUMMARY] mean val_mae = {np.mean(maes):.4f} ± {np.std(maes):.4f}")
    print(f"[CV SUMMARY] mean val_mape = {np.mean(mapes):.4f} ± {np.std(mapes):.4f}")
    print(f"[CV SUMMARY] mean val_mse = {np.mean(mses):.4f} ± {np.std(mses):.4f}")
    print(f"[CV SUMMARY] mean val_rmse = {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
    print(f"[CV SUMMARY] mean val_loss = {np.mean(losses):.4f} ± {np.std(losses):.4f}")
    print(f"[CV SUMMARY] mean val_auc  = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    return all_fold_results


if __name__ == "__main__":
    DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    random_seed(args.seed)
    g_train = torch.Generator().manual_seed(args.seed)
    g_val = torch.Generator().manual_seed(args.seed + 1)
    start_time = time.time()
    print(f"model training starts at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    df_valid = pd.read_csv(args.input)
    cv_validation(df_valid, args.pkl_dir, DEVICE)
    end_time = time.time()
    print(f"model training finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(time_since(start_time, end_time))

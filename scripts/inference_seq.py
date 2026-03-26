import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, f1_score, \
    precision_recall_curve, auc, matthews_corrcoef
from PANSY import TCR_pMHC_binding
from data import TotalDataset, collate_fn, tcr_pep_map, mhc_pep_map, get_global_feature, load_mhc_sequences



parser = argparse.ArgumentParser(description='predict whether the tcr and peptide can bind')
parser.add_argument('--input_file', type=str, default='data/test_seq/Unseen-TCR.csv',help='input CSV file')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--tcr_pmhc_model', type=str, default='checkpoints/PANSY.pt')
parser.add_argument('--output_dir', type=str, default='seq_outputs/')
parser.add_argument('--ppv_n', type=int, default=10, help='top-n value for PPVn calculation')
args = parser.parse_args()


os.makedirs(args.output_dir, exist_ok=True)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = TCR_pMHC_binding()
model.load_state_dict(torch.load(args.tcr_pmhc_model))
model.to(DEVICE)
model.eval()


def compute_ppv_n(probs, labels, n):
    df = pd.DataFrame({"prob": probs, "label": labels})
    df_sorted = df.sort_values(by="prob", ascending=False)
    df_topn = df_sorted.head(n)
    tp = df_topn["label"].sum()
    ppvn = tp / n
    return ppvn

def evaluate(model, test_loader):
    probs, preds, test_labels = [], [], []
    with torch.no_grad():
        for tcr_pep_maps, mhc_pep_maps, globals, labels in test_loader:
            tcr_pep_maps, mhc_pep_maps, globals, labels = (
                tcr_pep_maps.to(DEVICE), mhc_pep_maps.to(DEVICE),
                globals.to(DEVICE), labels.to(DEVICE))
            outputs = model(tcr_pep_maps, mhc_pep_maps, globals)
            batch_probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().tolist()
            batch_preds = [1 if p > 0.5 else 0 for p in batch_probs]
            probs.extend(batch_probs)
            preds.extend(batch_preds)
            test_labels.extend(labels.tolist())
    ACC = accuracy_score(test_labels, preds)
    ROC_AUC = roc_auc_score(test_labels, probs)
    Recall = recall_score(test_labels, preds)
    Precision = precision_score(test_labels, preds)
    F1 = f1_score(test_labels, preds)
    precision_curve, recall_curve, _ = precision_recall_curve(test_labels, probs)
    PR_AUC = auc(recall_curve, precision_curve)
    MCC = matthews_corrcoef(test_labels, preds)
    return probs, preds, test_labels, ACC, ROC_AUC, Recall, Precision, F1, PR_AUC, MCC



input_df = pd.read_csv(args.input_file)
cdr3s = input_df['CDR3'].tolist()
peptides = input_df['epitope'].tolist()
mhcs = load_mhc_sequences(input_df['MHC'].tolist())
labels = input_df['label'].tolist()
tcr_pep_data = tcr_pep_map(cdr3s, peptides)
mhc_pep_data = mhc_pep_map(mhcs, peptides)
global_data = get_global_feature(cdr3s, peptides, mhcs)
dataset = TotalDataset(tcr_pep_data, mhc_pep_data, global_data, labels)
test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
probs, preds, test_labels, ACC, ROC_AUC, Recall, Precision, F1, PR_AUC, MCC = evaluate(model, test_loader)
input_df['probs'] = probs
input_df['preds'] = preds
output_path = os.path.join(args.output_dir, 'predicted_results.csv')
input_df.to_csv(output_path, index=False)
ppvn_result = None
if args.ppv_n is not None:
    n = min(args.ppv_n, len(probs))
    ppvn_result = compute_ppv_n(probs, test_labels, n)
    print(f"PPV@{n}: {ppvn_result:.4f}")

summary = {
    "ACC": ACC,
    "ROC_AUC": ROC_AUC,
    "Recall": Recall,
    "Precision": Precision,
    "F1": F1,
    "PR_AUC": PR_AUC,
    "MCC": MCC,
    "PPV_n": ppvn_result,
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(args.output_dir, 'results_summary.csv'), index=False)
print("=== Evaluation Completed ===")
print(summary)

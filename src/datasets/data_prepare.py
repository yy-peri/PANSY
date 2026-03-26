import argparse
import os
from utils import random_seed
import pandas as pd
from data import TotalDataset, get_global_feature, tcr_pep_map, mhc_pep_map, load_mhc_sequences
import torch
import random
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='train the TCR_pMHC prediction model')
# file dir
parser.add_argument('--input', type=str, default='./data/train.csv',
                    help='path to input data, includes the following columns: CDR3, MHC, epitope')
parser.add_argument('--train_data', type=str, default='./data/train_cache/', help='path to save processed train data')
parser.add_argument('--val_data', type=str, default='./data/train_cache/', help='path to save processed val data')
parser.add_argument('--neg_mode', type=str, default='Random_Shuffle', help='negative sampling')
parser.add_argument('--neg_num', type=int, default=1, help='number of negatives per positive')
parser.add_argument('--seed', type=int, default=918, help='random seed')
args = parser.parse_args()

random_seed(args.seed)


def build_pos_triplets(tcrs, peptides, mhcs):
    return set(zip(tcrs, peptides, mhcs))


# random_shuffle
def neg_random_shuffle(tcrs, peptides, mhcs, pos_triplets, num_per_pos=1):
    all_peps_unique = list(set(peptides))
    all_mhcs_unique = list(set(mhcs))
    neg_triples_set = set()
    target_count = len(tcrs) * num_per_pos
    max_attempts = target_count * 10
    attempts = 0
    while len(neg_triples_set) < target_count and attempts < max_attempts:
        tcr = random.choice(tcrs)
        neg_pep = random.choice(all_peps_unique)
        neg_mhc = random.choice(all_mhcs_unique)
        new_triplet = (tcr, neg_pep, neg_mhc)
        if new_triplet not in pos_triplets:
            neg_triples_set.add(new_triplet)
        attempts += 1
    if len(neg_triples_set) < target_count:
        print(
            f"Warning: [neg_random_shuffle] Unable to generate a sufficient number of unique negative samples. Target: {target_count}, Actual count generated: {len(neg_triples_set)}.")
    return list(neg_triples_set)


# ===== make_data =====
def make_data(tcrs, peptides, mhcs, mode=None, neg_num=1):
    print(f"Creating data with mode: {mode}")
    pos_tcr_pep_map = tcr_pep_map(tcrs, peptides)
    pos_mhc_pep_map = mhc_pep_map(mhcs, peptides)
    pos_global_data = get_global_feature(tcrs, peptides, mhcs)
    pos_triplets = build_pos_triplets(tcrs, peptides, mhcs)
    neg_list = neg_random_shuffle(tcrs, peptides, mhcs, pos_triplets, num_per_pos=neg_num)
    samples = [(t, p, m, 1) for t, p, m in zip(tcrs, peptides, mhcs)] + \
              [(t, p, m, 0) for t, p, m in neg_list]
    neg_tcr_pep_map_list, neg_mhc_pep_map_list, neg_global_list = [], [], []
    for t, p, m in neg_list:
        neg_tcr_pep_map_list.append(tcr_pep_map([t], [p]))
        neg_mhc_pep_map_list.append(mhc_pep_map([m], [p]))
        neg_global_list += get_global_feature([t], [p], [m])
    neg_tcr_pep_map = torch.cat(neg_tcr_pep_map_list, dim=0)
    neg_mhc_pep_map = torch.cat(neg_mhc_pep_map_list, dim=0)
    total_tcr_pep_map = torch.cat((pos_tcr_pep_map, neg_tcr_pep_map), dim=0)
    total_mhc_pep_map = torch.cat((pos_mhc_pep_map, neg_mhc_pep_map), dim=0)
    total_global_data = pos_global_data + neg_global_list
    labels = [1] * len(pos_global_data) + [0] * len(neg_global_list)
    dataset = TotalDataset(total_tcr_pep_map, total_mhc_pep_map, total_global_data, labels)
    print(f"pos: {len(pos_global_data)}, neg: {len(neg_global_list)}, total: {len(labels)}")
    return dataset, samples


def save_processed_data(tcrs, peptides, mhcs, base_dir, mode=None, file_name=None, neg_num=1):
    save_dir = os.path.join(base_dir, mode)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, file_name)
    dataset, samples = make_data(tcrs, peptides, mhcs, mode=mode, neg_num=neg_num)
    torch.save({
        'tcr_pep_map': dataset.tcr_pep_map.cpu(),
        'mhc_pep_map': dataset.mhc_pep_map.cpu(),
        'graphs': [g.cpu() for g in dataset.graphs],
        'labels': dataset.labels
    }, save_path)
    df_samples = pd.DataFrame(samples, columns=['CDR3', 'epitope', 'MHC', 'label'])
    csv_path = save_path + '_all.csv'
    df_samples.to_csv(csv_path, index=False)


if __name__ == '__main__':
    df = pd.read_csv(args.input)
    cdr3s = df['CDR3'].tolist()
    peptides = df['epitope'].tolist()
    mhcs = load_mhc_sequences(df['MHC'].tolist())
    train_tcrs, val_tcrs, train_peptides, val_peptides, train_mhcs, val_mhcs = train_test_split(
        cdr3s, peptides, mhcs, test_size=0.1, random_state=0)
    save_processed_data(train_tcrs, train_peptides, train_mhcs, base_dir=args.train_data, mode=args.neg_mode, file_name='train_data', neg_num=args.neg_num)
    save_processed_data(val_tcrs, val_peptides, val_mhcs, base_dir=args.val_data, mode=args.neg_mode, file_name='val_data', neg_num=args.neg_num)
    print('data saved')

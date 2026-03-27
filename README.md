# A biophysically informed, interface-coupled framework for interpretable TCR–pMHC recognition
This repository provides the implementation of PANSY.
It includes both the seq branch and the res branch.
![](./output/overview.jpg)


PANSY is a framework designed for TCR–epitope binding prediction with two complementary branches: a sequence-based branch (`seq`) and a structure/residue-related branch (`res`). This repository provides the implementation for both branches, including data preparation, model training, and inference pipelines.

## Installation
### 1. Create a virtual environment
```bash
conda create -n PANSY python=3.10
conda activate PANSY
```
### 2. Install dependencies
Install all required packages using the provided `requirements.txt` file.
```bash
pip install -r requirements.txt
```
Main dependencies
- `torch==1.12.1`
- `numpy==1.26.4`
- `pandas==2.2.2`
- `scikit-learn==1.5.1`
- `tqdm==4.66.5`
- `peptides==0.3.4`
- `mhcnames==0.4.8`
- `torch-geometric==2.3.1`
- `timm==1.0.15`


**Note:** If you plan to use GPU acceleration, please make sure that your CUDA and cuDNN versions are compatible with your installed PyTorch version. Please refer to the official [PyTorch](https://pytorch.org/) website for version guidance.

**Note:** Please make sure that your Python environment is activated before running the installation command.

## PANSY-seq
### 1. Data Preparation
The input training file should be a CSV file containing **TCR–pMHC triples** with the following columns:

| Column | Description |
|--------|-------------|
| `CDR3` | CDR3 sequence of the TCR β-chain |
| `epitope` | Amino acid sequence of the epitope peptide |
| `MHC` | MHC allele name |

Example (`CSV` format):
```csv
CDR3,epitope,MHC
CASSFEAGQGFFSNQPQHF,FLKEKGGL,HLA-B*08:01
CASSTTSRGAISTDTQYF,LLYDANYFL,HLA-A*02:01
```
Before training, please preprocess the raw data into cached files. This avoids recomputing TCR–epitope maps, MHC–epitope maps, and global graph features in repeated experiments.

Run
```bash
python -m src.datasets.data_prepare \
  --input data/train.csv \
  --train_data data/train_cache/ \
  --val_data data/train_cache/ \
  --neg_mode Random_Shuffle \
  --neg_num 1 \
  --seed 918
```

**Note:** The script saves the processed training and validation cache files under the negative-sampling subdirectory, e.g., `data/train_cache/Random_Shuffle/train_data` and `data/train_cache/Random_Shuffle/val_data`.

### 2. Training
After preparing the cached data, you can train the `PANSY-seq` model using `scripts/train_seq.py`.

Run
```bash
python scripts/train_seq.py \
  --train_data data/train_cache/Random_Shuffle/train_data \
  --val_data data/train_cache/Random_Shuffle/val_data \
  --model_dir checkpoints/PANSY-seq.pt \
  --batch_size 512 \
  --lr 5e-4 \
  --max_epoch 500 \
  --seed 918
```
The trained model checkpoint will be saved under the `checkpoints/` directory. The released checkpoint for this branch is `checkpoints/PANSY-seq.pt`.

### 3. Inference
After training, or by directly using a provided checkpoint, you can perform inference with `scripts/inference_seq.py`.

Run
```bash
python scripts/inference_seq.py \
  --input_file data/test_seq/Unseen-TCR.csv \
  --batch_size 64 \
  --tcr_pmhc_model checkpoints/PANSY-seq.pt \
  --output_dir seq_outputs/ \
  --ppv_n 10
```
The inference script loads the trained `PANSY-seq` checkpoint from `checkpoints/PANSY-seq.pt` and performs prediction on the sequence branch input data.

## PANSY-res
### 1. Data Preparation
The input training file should be a CSV file containing **TCR–pMHC structural samples** with the following columns:

| Column | Description |
|--------|-------------|
| `PDB` | Structure identifier used to retrieve the corresponding residue-level supervision file |
| `CDR3` | CDR3 sequence of the TCR β-chain |
| `epitope` | Amino acid sequence of the epitope peptide |
| `MHC` | MHC allele name |

Example (`CSV` format):
```csv
PDB,CDR3,epitope,MHC
8gom,CASTWGRASTDTQYF,RLQSLQTYV,HLA-A*02:01
7n1e,CASSLGGAGGADTQYF,RLQSLQTYV,HLA-A*02:01
```
In addition to the CSV file, `PANSY-res` requires a directory containing one residue-level annotation file for each structure. Each sample in the CSV file is matched to a `.pkl` file using the value in the `PDB` column. The file name should therefore follow the format:
```text
{PDB}.pkl
```
For example:
```text
8gom.pkl
7n1e.pkl
```
Each `.pkl` file should provide residue-level supervision for the corresponding structure. In the current implementation, the loader reads the `cdr3_beta` entry and expects at least the following fields:
- `dist`: residue-wise distance matrix between CDR3β residues and epitope residues
- `contact`: residue-wise binary contact matrix between CDR3β residues and epitope residues

### 2. Training
After preparing the CSV file and the corresponding residue-level supervision files, you can train the `PANSY-res` model using `scripts/train_res.py`.

Run
```bash
python scripts/train_res.py \
  --input data/structure.csv \
  --pkl_dir data/structure_data/structure \
  --output_dir res-outputs/ \
  --model_dir checkpoints/ \
  --pretrained_ckpt checkpoints/PANSY-seq.pt \
  --num_folds 10 \
  --batch_size 16 \
  --lr 5e-3 \
  --max_epoch 100 \
  --seed 918 \
  --split_mode random \
  --finetune_mode all
```

**Note:** In the current implementation, the released sequence-level checkpoint `checkpoints/PANSY-seq.pt` is used as the initialization for residue-level training.

### 3. Inference
After training, or by directly using a provided checkpoint, you can perform residue-level inference with `scripts/inference_res.py`.

Run
```bash
python scripts/inference_res.py \
  --input_csv data/prediction.csv \
  --ckpt checkpoints/PANSY-res.pt \
  --out_npz res-outputs/output.npz \
  --batch_size 64 \
  --seed 918 \
  --device cuda:0
```

The inference script loads the trained `PANSY-res` checkpoint from checkpoints/PANSY-res.pt and predicts residue-level distance maps and contact probability maps for the input TCR–pMHC samples.




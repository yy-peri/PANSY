import argparse
import random
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from src.utils import EarlyStopping, time_since, random_seed
from torch.utils.data import DataLoader
from src.seq.PANSY_seq import TCR_pMHC_binding
from src.datasets.dataset_seq import TotalDataset, collate_fn



parser = argparse.ArgumentParser(description='train the TCR_pMHC prediction model')
parser.add_argument('--train_data', type=str, default='data/train_cache/train_data', help='path to input data,\
includes the following three columns:CDR3, MHC, epitope')
parser.add_argument('--val_data', type=str, default='data/train_cache/val_data', help='path to input data,\
includes the following three columns:CDR3, MHC, epitope')
parser.add_argument('--model_dir', type=str, default='checkpoints/PANSY.pt', help='where to save model')
parser.add_argument('--batch_size', type=int, default=512, help='batch_size for tcr_pmhc prediction model')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--max_epoch', type=int, default=500, help='max epoch')
parser.add_argument('--seed', type=int, default=918, help='random seed')
args = parser.parse_args()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def load_cached_dataloader(path, batch_size, train=True):
    data = torch.load(path)
    dataset = TotalDataset(
        data['tcr_pep_map'],
        data['mhc_pep_map'],
        data['graphs'],
        data['labels']
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=16,
        worker_init_fn=seed_worker,
        generator=(g_train if train else g_val),
        collate_fn=collate_fn
    )


# model training
def train(epoch):
    model.train()
    train_loss = 0.0
    for tra_step, (tcr_pep_maps, mhc_pep_maps, globals, labels) in enumerate(train_dataloader, 1):
        tcr_pep_maps, mhc_pep_maps, globals, labels = (tcr_pep_maps.to(DEVICE), mhc_pep_maps.to(DEVICE),
                                                       globals.cpu(), labels.to(DEVICE))
        outputs = model(tcr_pep_maps, mhc_pep_maps, globals)
        loss = Loss(outputs, labels)
        # calculate the accuracy
        probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        preds = [1 if prob > 0.5 else 0 for prob in probs]
        accuracy = accuracy_score(labels.detach().cpu().numpy(), preds)
        train_loss += loss.detach()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if tra_step % 60 == 0:
            print(f'epoch: {epoch}, tra_step: {tra_step}, train_loss: {train_loss/tra_step}, loss: {loss.detach():.3f}, accuracy: {accuracy:.3f}')

def validation(epoch):
    model.eval()
    val_loss = 0.0
    all_step = 0
    with torch.no_grad():
        for val_step, (tcr_pep_maps, mhc_pep_maps, globals, labels) in enumerate(val_dataloader, 1):
            tcr_pep_maps, mhc_pep_maps, globals, labels = (tcr_pep_maps.to(DEVICE), mhc_pep_maps.to(DEVICE),
                                                           globals.cpu(), labels.to(DEVICE))
            outputs = model(tcr_pep_maps, mhc_pep_maps, globals)
            loss = Loss(outputs, labels)
            val_loss += loss.detach()
            all_step = val_step
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            preds = [1 if prob > 0.5 else 0 for prob in probs]
            accuracy = accuracy_score(labels.detach().cpu().numpy(), preds)
            if val_step % 6 == 0:
                print(f'epoch: {epoch}, val_step: {val_step}, val_loss: {val_loss/val_step}, loss: {loss.detach():.3f}, accuracy: {accuracy:.3f}')
    return val_loss/all_step


if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    random_seed(args.seed)
    g_train = torch.Generator().manual_seed(args.seed)
    g_val = torch.Generator().manual_seed(args.seed + 1)
    train_dataloader = load_cached_dataloader(args.train_data, args.batch_size, train=True)
    val_dataloader = load_cached_dataloader(args.val_data, args.batch_size, train=False)

    # Initialize tcr-pmhc prediction model
    model = TCR_pMHC_binding()
    model.to(DEVICE)
    model.graph_encoder.cpu()

    Loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    early_stopping = EarlyStopping(patience=6, verbose=True, save_path=args.model_dir, mode='min', monitor_name='val_loss')


    start_time = time.time()
    print(f'model training starts at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')
    for epoch in range(1, args.max_epoch+1):
        train(epoch)
        val_loss = validation(epoch)
        lr_scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            break
        # save the model
    torch.save(model.state_dict(), args.model_dir)  
    end_time = time.time()
    print(f'model training finished at: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))}')
    print(time_since(start_time, end_time))

















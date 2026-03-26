import random
import os
import math
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import median_absolute_error, roc_auc_score, matthews_corrcoef

# set the random seeds
def random_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, save_path='.', delta=0.0, mode='min', monitor_name='metric'):
        assert mode in ['min', 'max']
        self.patience = patience
        self.verbose = verbose
        self.save_path = save_path
        self.delta = delta
        self.mode = mode
        self.monitor_name = monitor_name
        self.best_score = None
        self.best_value = np.Inf if mode == 'min' else -np.Inf
        self.counter = 0
        self.early_stop = False
    def _is_better(self, score, best_score):
        if self.mode == 'min':
            return score < (best_score - self.delta)
        else:  # 'max'
            return score > (best_score + self.delta)
    def __call__(self, current_value, model):
        score = current_value
        if self.best_score is None:
            self._save_checkpoint(prev_best=self.best_value, current_value=current_value, model=model, first=True)
            self.best_score = score
            self.best_value = current_value
            return
        if self._is_better(score, self.best_score):
            self._save_checkpoint(prev_best=self.best_value, current_value=current_value, model=model, first=False)
            self.best_score = score
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, '
                  f'{self.monitor_name}: {current_value:.6f}')
            if self.counter >= self.patience:
                self.early_stop = True
    def _save_checkpoint(self, prev_best, current_value, model, first=False):
        if self.verbose:
            if first:
                print(f'{self.monitor_name} initial save. Saving model ... (best: {prev_best:.6f} -> {current_value:.6f})')
            else:
                print(f'{self.monitor_name} improved. Saving model ... (best: {prev_best:.6f} -> {current_value:.6f})')
        torch.save(model.state_dict(), self.save_path)


def time_since(start, end):
    s = end - start
    m = math.floor(s/60)
    s -= m * 60
    return f'total time:{m}m {s}s'



def get_scores_dist(y_true, y_pred, y_mask, mape_eps=1e-8):
    """
    Returns:
      avg_metrics_dist = [avg_pearson, avg_medae, avg_mape, avg_mse, avg_rmse]
      metrics_dist = [pearson_list, medae_list, mape_list, mse_list, rmse_list]
    """
    y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
    y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else np.array(y_pred)
    y_mask = y_mask.detach().cpu().numpy() if torch.is_tensor(y_mask) else np.array(y_mask)
    coef_list, mae_list, mape_list = [], [], []
    mse_list, rmse_list = [], []
    for yt, yp, ym in zip(y_true, y_pred, y_mask):
        ym = ym.astype(bool)
        yt = yt[ym]
        yp = yp[ym]
        # Pearson
        try:
            coef, _ = pearsonr(yt, yp)
        except Exception:
            coef = np.nan
        coef_list.append(coef)
        try:
            mae = median_absolute_error(yt, yp)
        except Exception:
            mae = np.nan
        mae_list.append(mae)
        denom = np.maximum(np.abs(yt), mape_eps)
        mape = np.median(np.abs((yt - yp) / denom)) if yt.size > 0 else np.nan
        mape_list.append(mape)
        if yt.size > 0:
            diff = yt - yp
            mse = np.mean(diff * diff)
            rmse = np.sqrt(mse)
        else:
            mse = np.nan
            rmse = np.nan
        mse_list.append(mse)
        rmse_list.append(rmse)
    avg_coef = np.nanmean(coef_list)
    avg_mae = np.nanmean(mae_list)
    avg_mape = np.nanmean(mape_list)
    avg_mse = np.nanmean(mse_list)
    avg_rmse = np.nanmean(rmse_list)
    return [avg_coef, avg_mae, avg_mape, avg_mse, avg_rmse], \
           [coef_list, mae_list, mape_list, mse_list, rmse_list]


def get_scores_contact(y_true, y_pred_prob, y_mask, thr=0.5):
    """
    Returns:
      avg_metrics_contact = [avg_auc, avg_mcc]
      metrics_contact = [auc_list, mcc_list]
    """
    y_true = y_true.detach().cpu().numpy() if torch.is_tensor(y_true) else np.array(y_true)
    y_pred = y_pred_prob.detach().cpu().numpy() if torch.is_tensor(y_pred_prob) else np.array(y_pred_prob)
    y_mask = y_mask.detach().cpu().numpy() if torch.is_tensor(y_mask) else np.array(y_mask)
    auc_list = []
    mcc_list = []
    for yt, yp, ym in zip(y_true, y_pred, y_mask):
        ym = ym.astype(bool)
        yt = yt[ym]
        yp = yp[ym]
        try:
            auc = roc_auc_score(yt, yp)
        except Exception:
            auc = np.nan
        auc_list.append(auc)
        try:
            y_hat = (yp >= thr).astype(int)
            if len(np.unique(yt)) < 2 or len(np.unique(y_hat)) < 2:
                mcc = np.nan
            else:
                mcc = matthews_corrcoef(yt, y_hat)
        except Exception:
            mcc = np.nan
        mcc_list.append(mcc)
    avg_auc = np.nanmean(auc_list)
    avg_mcc = np.nanmean(mcc_list)
    return [avg_auc, avg_mcc], [auc_list, mcc_list]


def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.Conv3d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose1d):
        torch.nn.init.normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.ConvTranspose3d):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.BatchNorm3d):
        torch.nn.init.normal_(m.weight.data, mean=1, std=0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)
    elif isinstance(m, torch.nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, torch.nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, torch.nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)
    elif isinstance(m, torch.nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param.data)
            else:
                torch.nn.init.normal_(param.data)



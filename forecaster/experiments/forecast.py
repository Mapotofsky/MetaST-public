import os
from os.path import join
import math
import logging
from typing import Callable, Optional, Union, Dict, Tuple
from tqdm import tqdm

import gin
from fire import Fire
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.base import Experiment
from experiments.datasets import ColdStartForecastDataset
from models import get_model
from utils.checkpoint import Checkpoint
from utils.ops import default_device, to_tensor
from utils.losses import get_loss_fn
from utils.metrics import calc_metrics


class ForecastExperiment(Experiment):
    def fix_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @gin.configurable()
    def instance(self,
                 model_type: str,
                 seed: int = None,
                 save_vals: Optional[bool] = True,):
        self.seed = seed
        if self.seed is not None:
            self.fix_seed(self.seed)

        # load datasets, model, checkpointer
        train_set, train_loader = get_data(flag='train', seed=self.seed)
        val_set, val_loader = get_data(flag='val', seed=self.seed)
        test_set, test_loader = get_data(flag='test', seed=self.seed)

        mask_spectrum = None
        if model_type == 'koopa':
            amps = 0.0
            for data in train_loader:
                lookback_window = data[0]  # x
                lookback_window = lookback_window[..., 0]
                amps += abs(torch.fft.rfft(lookback_window, dim=1)).mean(dim=0).mean(dim=1)

            mask_spectrum = amps.topk(int(amps.shape[0] * 0.2)).indices

        device = default_device()
        model = get_model(model_type, device=device, mask_spectrum=mask_spectrum).to(device)
        checkpoint = Checkpoint(self.root)

        # train forecasting task
        model = train(model, checkpoint, train_loader, val_loader, test_loader)

        # testing
        val_metrics_seen, val_metrics_unseen, val_metrics = validate(model, loader=val_loader, report_metrics=True)
        test_metrics_seen, test_metrics_unseen, test_metrics = validate(model, loader=test_loader, report_metrics=True,
                                                                        save_path=self.root if save_vals else None)
        np.save(join(self.root, 'metrics.npy'), {'val_seen': val_metrics_seen, 'test_seen': test_metrics_seen,
                                                 'val_unseen': val_metrics_unseen, 'test_unseen': test_metrics_unseen,
                                                 'val': val_metrics, 'test': test_metrics})

        val_metrics = {f'ValMetric/{k}': v for k, v in val_metrics.items()}
        test_metrics = {f'TestMetric/{k}': v for k, v in test_metrics.items()}
        checkpoint.close({**val_metrics, **test_metrics})


@gin.configurable()
def get_optimizer(model: nn.Module,
                  lr: Optional[float] = 1e-3,
                  lambda_lr: Optional[float] = 1.,
                  weight_decay: Optional[float] = 1e-2) -> optim.Optimizer:
    group1 = []  # lambda
    group2 = []  # no decay
    group3 = []  # decay
    no_decay_list = ('bias', 'norm',)
    for param_name, param in model.named_parameters():
        if '_lambda' in param_name:
            group1.append(param)
        elif any([mod in param_name for mod in no_decay_list]):
            group2.append(param)
        else:
            group3.append(param)
    optimizer = optim.Adam([
        {'params': group1, 'weight_decay': 0, 'lr': lambda_lr, 'scheduler': 'cosine_annealing'},
        {'params': group2, 'weight_decay': 0, 'scheduler': 'cosine_annealing_with_linear_warmup'},
        {'params': group3, 'scheduler': 'cosine_annealing_with_linear_warmup'}
    ], lr=lr, weight_decay=weight_decay)
    return optimizer


@gin.configurable()
def get_scheduler(optimizer: optim.Optimizer,
                  T_max: int,
                  warmup_epochs: int,
                  eta_min: Optional[float] = 0.) -> optim.lr_scheduler.LambdaLR:
    scheduler_fns = []
    for param_group in optimizer.param_groups:
        scheduler = param_group['scheduler']
        if scheduler == 'none':
            fn = lambda T_cur: 1
        elif scheduler == 'cosine_annealing':
            lr = eta_max = param_group['lr']
            fn = lambda T_cur: (eta_min + 0.5 * (eta_max - eta_min) * (
                1.0 + math.cos((T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
        elif scheduler == 'cosine_annealing_with_linear_warmup':
            lr = eta_max = param_group['lr']
            # https://blog.csdn.net/qq_36560894/article/details/114004799
            fn = lambda T_cur: T_cur / warmup_epochs if T_cur < warmup_epochs else (eta_min + 0.5 * (
                eta_max - eta_min) * (1.0 + math.cos(
                    (T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
        else:
            raise ValueError(f'No such scheduler, {scheduler}')
        scheduler_fns.append(fn)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fns)
    return scheduler


@gin.configurable()
def get_data(flag: bool,
             seed: int,
             batch_size: int) -> Tuple[ColdStartForecastDataset, DataLoader]:
    if flag in ('val', 'test'):
        shuffle = False
        drop_last = False
    elif flag == 'train':
        shuffle = True
        drop_last = True
    else:
        raise ValueError(f'no such flag {flag}')
    dataset = ColdStartForecastDataset(flag, seed)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=drop_last)
    return dataset, data_loader


@gin.configurable()
def train(model: nn.Module,
          checkpoint: Checkpoint,
          train_loader: DataLoader,
          val_loader: DataLoader,
          test_loader: DataLoader,
          loss_name: str,
          epochs: int,
          max_nodes_per_batch: Optional[int] = None) -> nn.Module:
    """
    Training phase: Get seen_nodes data from dataloader -> Iterate through node_id to select target_node
                    -> Mask target_node part in seen_nodes -> Input two parts to model
                    -> Get prediction output for target_node -> Concatenate all nodes, calculate overall loss
    """

    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer=optimizer, T_max=epochs)
    training_loss_fn = get_loss_fn(loss_name)

    for epoch in range(epochs):
        train_loss = []
        model.train()
        for it, data in enumerate(train_loader):
            optimizer.zero_grad()

            x_seen, _, y_seen, _, x_time, y_time = map(to_tensor, data)
            n_nodes = x_seen.shape[-2]  # (B, L, N', C)

            # If max nodes per batch is set and current node count exceeds limit, randomly sample
            if max_nodes_per_batch is not None and n_nodes > max_nodes_per_batch:
                # Randomly select node indices
                selected_indices = torch.randperm(n_nodes)[:max_nodes_per_batch]
                selected_indices = selected_indices.sort()[0]
                y_seen_selected = y_seen[:, :, selected_indices]

                forecast = []
                for i, target_idx in enumerate(selected_indices):
                    forecast.append(model(x_seen, x_seen[:, :, target_idx, :], target_idx, x_time, y_time))  # (B, L, C) -> (B, L, )
                forecast = torch.stack(forecast, dim=-1)  # (B, L, max_nodes_per_batch)

                # Calculate loss using selected nodes
                loss = training_loss_fn(forecast, y_seen_selected)
            else:
                # Use all nodes
                forecast = []
                for target_idx in range(n_nodes):
                    forecast.append(model(x_seen, x_seen[:, :, target_idx, :], target_idx, x_time, y_time))  # (B, L, C) -> (B, L, )
                forecast = torch.stack(forecast, dim=-1)  # (B, L, N')

                loss = training_loss_fn(forecast, y_seen)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if (it + 1) % 100 == 0:
                logging.info(f"epochs: {epoch + 1}, iters: {it + 1} | training loss: {loss.item():.2f}")
        scheduler.step()

        train_loss = np.average(train_loss)
        # _, train_unseen_loss, train_all_loss = validate(model, loader=train_loader, loss_fn=training_loss_fn)
        val_seen_loss, val_unseen_loss, val_loss = validate(model, loader=val_loader, loss_fn=training_loss_fn)
        test_seen_loss, test_unseen_loss, test_loss = validate(model, loader=test_loader, loss_fn=training_loss_fn)

        scalars = {'Loss/Train': train_loss,
                   # 'Loss/Train_unseen': train_unseen_loss,
                   # 'Loss/Train_all': train_all_loss,
                   'Loss/Val_seen': val_seen_loss,
                   'Loss/Val_unseen': val_unseen_loss,
                   'Loss/Val': val_loss,
                   'Loss/Test_seen': test_seen_loss,
                   'Loss/Test_unseen': test_unseen_loss,
                   'Loss/Test': test_loss}
        for i, param_group in enumerate(optimizer.param_groups):
            scalars[f'LR/group_{i}'] = param_group['lr']
        checkpoint(epoch + 1, model, scalars=scalars)

        torch.cuda.empty_cache()

        if checkpoint.early_stop:
            logging.info("Early stopping")
            break

    if epochs > 0:
        model.load_state_dict(torch.load(checkpoint.model_path))
    return model


@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             loss_fn: Optional[Callable] = None,
             report_metrics: Optional[bool] = False,
             save_path: Optional[str] = None) -> Union[Dict[str, float], float]:
    """
    Inference phase: Get seen_nodes and unseen_nodes data from dataloader -> Process seen same as training
                    -> Calculate seen_nodes loss separately -> Iterate through unseen_nodes to select target_node
                    -> Input seen_nodes and target_node to model to get prediction output
                    -> Concatenate and calculate unseen_nodes loss -> Calculate total loss
    """
    model.eval()
    preds = []
    trues = []
    inps = []
    total_loss_seen = []
    total_loss_unseen = []
    total_loss = []
    for it, data in enumerate(loader):
        x_seen, x_unseen, y_seen, y_unseen, x_time, y_time = map(to_tensor, data)
        n_seen_nodes, n_unseen_nodes = y_seen.shape[2], y_unseen.shape[2]

        forecast = []
        for target_idx in range(n_seen_nodes):  # (B, L, N', C)
            forecast.append(model(x_seen, x_seen[:, :, target_idx, :], target_idx, x_time, y_time))  # (B, L, C) -> (B, L, )
        for target_idx in range(n_unseen_nodes):  # (B, L, N'', C)
            forecast.append(model(x_seen, x_unseen[:, :, target_idx, :], None, x_time, y_time))  # (B, L, C) -> (B, L, )
        forecast = torch.stack(forecast, dim=2)  # (B, L, N'')

        x = torch.cat([x_seen, x_unseen], dim=2)  # (B, L, N, C)
        y = torch.cat([y_seen, y_unseen], dim=2)  # (B, L, N)
        # Follow BasicTS recommendation, perform inverse normalization
        if loader.dataset.scale:
            forecast_np = forecast.detach().cpu().numpy()
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            # Avoid floating point errors
            forecast_inv = loader.dataset.inverse_transform(forecast_np)
            x_inv = np.round(loader.dataset.inverse_transform(x_np[..., 0]))
            y_inv = np.round(loader.dataset.inverse_transform(y_np))
            forecast = torch.from_numpy(forecast_inv).to(forecast.device)
            x = torch.from_numpy(x_inv).to(x_seen.device)
            y = torch.from_numpy(y_inv).to(y_seen.device)

        if report_metrics:
            preds.append(forecast)
            trues.append(y)
            if save_path is not None:
                inps.append(x)
        else:
            loss_seen = loss_fn(forecast[:, :, :n_seen_nodes], y[:, :, :n_seen_nodes], reduction='none')
            loss_unseen = loss_fn(forecast[:, :, -n_unseen_nodes:], y[:, :, -n_unseen_nodes:], reduction='none')
            loss = loss_fn(forecast, y, reduction='none')

            total_loss_seen.append(loss_seen)
            total_loss_unseen.append(loss_unseen)
            total_loss.append(loss)

    if report_metrics:
        preds = torch.cat(preds, dim=0).detach().cpu().numpy()
        trues = torch.cat(trues, dim=0).detach().cpu().numpy()
        if save_path is not None:
            inps = torch.cat(inps, dim=0).detach().cpu().numpy()
            np.savez(join(save_path, 'inps.npz'), seen=inps[:, :, :n_seen_nodes], unseen=inps[:, :, -n_unseen_nodes:])
            np.savez(join(save_path, 'preds.npz'), seen=preds[:, :, :n_seen_nodes], unseen=preds[:, :, -n_unseen_nodes:])
            np.savez(join(save_path, 'trues.npz'), seen=trues[:, :, :n_seen_nodes], unseen=trues[:, :, -n_unseen_nodes:])
        metrics_seen = calc_metrics(preds[:, :, :n_seen_nodes], trues[:, :, :n_seen_nodes])
        metrics_unseen = calc_metrics(preds[:, :, -n_unseen_nodes:], trues[:, :, -n_unseen_nodes:])
        metrics = calc_metrics(preds, trues)
        return metrics_seen, metrics_unseen, metrics

    total_loss_seen = torch.cat(total_loss_seen, dim=0).cpu()
    total_loss_unseen = torch.cat(total_loss_unseen, dim=0).cpu()
    total_loss = torch.cat(total_loss, dim=0).cpu()
    return np.average(total_loss_seen), np.average(total_loss_unseen), np.average(total_loss)


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)

    # If no arguments passed, batch execute command files
    if len(sys.argv) == 1:
        import glob
        import contextlib

        command_files = glob.glob(os.path.join('experiment_storage', 'anchor_size', '**', 'command'), recursive=True)
        print(f"Found {len(command_files)} command files")

        for i, cmd_file in enumerate(tqdm(command_files)):
            print(f"\n[{i+1}/{len(command_files)}] Executing: {cmd_file}")

            with open(cmd_file, 'r') as f:
                cmd_content = f.read().strip()

            parts = cmd_content.split(' --config_path=')
            if len(parts) == 2:
                config_path = os.path.join(os.path.dirname(cmd_file), 'config.gin')
                log_file = os.path.join(os.path.dirname(cmd_file), 'instance.log')

                # Reset logging configuration to avoid file handle conflicts
                for handler in logging.root.handlers[:]:
                    logging.root.removeHandler(handler)

                # Redirect output to log file
                with open(log_file, 'a', encoding='utf-8') as f:
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        # Set parameters and execute
                        sys.argv = ['forecast.py', 'run', f'--config_path={config_path}']
                        try:
                            Fire(ForecastExperiment)
                            print("Execution completed")
                        except Exception as e:
                            print(f"Failed: {e}")
            else:
                print(f"Command format error: {cmd_content}")
    else:
        Fire(ForecastExperiment)

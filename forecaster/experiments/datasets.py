import os
from os.path import join
from typing import Optional, List, Tuple, Union

import gin
import json
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

import sys
TARGET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(TARGET_DIR)
os.chdir(TARGET_DIR)


@gin.configurable()
class ColdStartForecastDataset(Dataset):
    def __init__(self,
                 flag: str,
                 seed: int,
                 horizon_len: int,
                 scale: bool,
                 data_path: str,
                 root_path: Optional[str] = os.path.join(os.path.dirname(TARGET_DIR), 'datasets'),
                 lookback_len: Optional[int] = None,
                 lookback_aux_len: Optional[int] = 0,
                 lookback_mult: Optional[float] = None,
                 train_ratio: Optional[float] = 0.7,
                 test_ratio: Optional[float] = 0.2,
                 seen_node_ratio: Optional[float] = 0.1):
        """
        Training phase: Get seen_nodes data from dataloader -> Iterate through node_id to select target_node
                       -> Mask target_node part in seen_nodes -> Input two parts to model
                       -> Get prediction output for target_node -> Concatenate all nodes, calculate overall loss
        Inference phase: Get seen_nodes and unseen_nodes data from dataloader -> Process seen same as training
                        -> Calculate seen_nodes loss separately -> Iterate through unseen_nodes to select target_node
                        -> Input seen_nodes and target_node to model to get prediction output
                        -> Concatenate and calculate unseen_nodes loss -> Calculate total loss
        :param flag: train/val/test flag
        :param horizon_len: number of time steps in forecast horizon
        :param scale: performs standard scaling
        :param data_path: relative (to root_path) path to data file (.csv)
        :param root_path: path to datasets folder
        :param lookback_len: number of time steps in lookback window
        :param lookback_aux_len: number of time steps to append to y from the lookback window
        (for models with decoders which requires initialisation)
        :param lookback_mult: multiplier to decide lookback window length
        """
        assert flag in ('train', 'val', 'test'), \
            "flag should be one of (train, val, test)"
        assert (lookback_len is not None) ^ (lookback_mult is not None), \
            "only 'lookback_len' xor 'lookback_mult' should be specified"
        assert seen_node_ratio > 0.0, "seen_node_ratio must be greater than 0"

        self.flag = flag
        self.seed = seed
        self.lookback_len = int(horizon_len * lookback_mult) if lookback_mult is not None else lookback_len
        self.lookback_aux_len = lookback_aux_len
        self.horizon_len = horizon_len
        self.scale = scale
        self.data_path = data_path
        self.root_path = root_path
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.seen_node_ratio = seen_node_ratio

        self.scaler = None
        self.data_x = None
        self.data_y = None
        self.timestamps = None
        self.n_time = None
        self.n_time_samples = None
        self.n_nodes = None
        self.n_features = None
        self.seen_nodes = None
        self.unseen_nodes = None
        self.mean = None
        self.std = None
        self.load_data()

    def load_data(self):
        description_file_path = join(self.root_path, f"{self.data_path}/desc.json")
        with open(description_file_path, 'r') as f:
            description = json.load(f)

        self.train_ratio = description['regular_settings']['TRAIN_VAL_TEST_RATIO'][0]
        self.test_ratio = description['regular_settings']['TRAIN_VAL_TEST_RATIO'][2]

        data_file_path = join(self.root_path, f"{self.data_path}/data.dat")
        ori_data = np.memmap(data_file_path, dtype='float32', mode='r', shape=tuple(description['shape']))
        ori_data = ori_data.copy()  # (T, N, C)
        n_timestamps, n_nodes, n_features = ori_data.shape
        self.n_nodes = n_nodes
        self.n_features = n_features

        self.get_nodes(self.seen_node_ratio)
        border1s, border2s, border1, border2 = self.get_borders(ori_data)

        data = ori_data.copy()
        # self.scaler = StandardScaler()
        # if self.scale:
        #     scale_data = ori_data[border1s[0]:border2s[0], :, 0]
        #     self.scaler.fit(scale_data)
        #     data[..., 0] = self.scaler.transform(ori_data[..., 0])
        if self.scale:
            scale_data = ori_data[border1s[0]:border2s[0], :, 0]
            self.mean = np.mean(scale_data)
            self.std = np.std(scale_data)
            if self.std == 0:
                self.std = 1.0
            data[..., 0] = (ori_data[..., 0] - self.mean) / self.std

        self.timestamps = np.empty((data[border1:border2].shape[0], 0))
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.n_time = len(self.data_x)
        self.n_time_samples = self.n_time - self.lookback_len - self.horizon_len + 1

    def get_nodes(self, seen_node_ratio: float):
        node_list = [i for i in range(self.n_nodes)]
        if seen_node_ratio < 1.0:
            n_seen_nodes = int(self.n_nodes * seen_node_ratio)
            n_unseen_nodes = self.n_nodes - n_seen_nodes
            node_path = join(self.root_path, f"{self.data_path}/node_list_{self.seed}.npy")
            if not os.path.exists(node_path):
                random.shuffle(node_list)
                node_list = np.array(node_list)
                np.save(node_path, node_list)
            else:
                node_list = np.load(node_path)
            self.seen_nodes = np.sort(node_list[:n_seen_nodes])  # for x, and y during training
            self.unseen_nodes = np.sort(node_list[-n_unseen_nodes:])  # for y during verification and testing
        else:
            # No missing nodes, train/test on all locations
            self.seen_nodes = np.array(node_list)
            self.unseen_nodes = np.array([])

    def get_borders(self, ori_data: pd.DataFrame) -> Tuple[List[int], List[int], List[int], List[int]]:
        set_type = {'train': 0, 'val': 1, 'test': 2}[self.flag]
        n_timestamps = ori_data.shape[0]

        num_train = int(n_timestamps * self.train_ratio)
        num_test = int(n_timestamps * self.test_ratio)
        num_val = n_timestamps - num_train - num_test
        border1s = [0, num_train - self.lookback_len, n_timestamps - num_test - self.lookback_len]
        border2s = [num_train, num_train + num_val, n_timestamps]

        border1 = border1s[set_type]
        border2 = border2s[set_type]
        return border1s, border2s, border1, border2

    def __len__(self):
        return self.n_time_samples

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_start = idx
        x_end = x_start + self.lookback_len
        y_start = x_end - self.lookback_aux_len
        y_end = y_start + self.lookback_aux_len + self.horizon_len

        x_seen = self.data_x[x_start:x_end, self.seen_nodes, :]  # (L, N', C)
        y_seen = self.data_y[y_start:y_end, self.seen_nodes, 0]  # (L, N') keep only monitoring data
        x_time = self.timestamps[x_start:x_end]  # (L, )
        y_time = self.timestamps[y_start:y_end]
        if len(self.unseen_nodes) == 0:
            x_unseen = np.empty((self.lookback_len, 0, self.n_features), dtype=self.data_x.dtype)
            y_unseen = np.empty((self.lookback_aux_len + self.horizon_len, 0), dtype=self.data_y.dtype)
        else:
            x_unseen = self.data_x[x_start:x_end, self.unseen_nodes, :]
            y_unseen = self.data_y[y_start:y_end, self.unseen_nodes, 0]

        return x_seen, x_unseen, y_seen, y_unseen, x_time, y_time

    def inverse_transform(self, data):
        # return self.scaler.inverse_transform(data)
        data = data * self.std + self.mean
        return data

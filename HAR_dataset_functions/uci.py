# datasets_uci.py

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class UCIHARDataset(Dataset):
    """
    UCI-HAR Dataset Loader
    반환:
        X_ct : (C, T)
        y    : int
        s    : subject ID
    """
    def __init__(self, data_path: str, split: str = 'train'):
        super().__init__()
        base = Path(data_path) / split

        # 1) 센서 9개(body_acc, total_acc, body_gyro × xyz)
        signals = []
        for sensor in ['body_acc', 'body_gyro', 'total_acc']:
            for axis in ['x', 'y', 'z']:
                file = base / 'Inertial Signals' / f'{sensor}_{axis}_{split}.txt'
                arr = np.loadtxt(file)
                signals.append(arr)

        # shape = (9, N, T)
        signals = np.array(signals)

        # (9, N, T) → (N, T, 9)
        self.X = np.transpose(signals, (1, 2, 0)).astype(np.float32)

        # 라벨 (1~6 → 0~5)
        y_path = base.parent / split / f'y_{split}.txt'
        self.y = np.loadtxt(y_path).astype(np.int64) - 1

        # subject
        try:
            subj_path = base.parent / split / f'subject_{split}.txt'
            self.subjects = np.loadtxt(subj_path).astype(np.int64)
        except:
            self.subjects = np.zeros(len(self.y), dtype=np.int64)

        print("=" * 80)
        print("[UCI-HAR] Loaded")
        print(f"  X shape : {self.X.shape} (N, T, C)")
        print(f"  y shape : {self.y.shape}")
        print(f"  subjects: unique = {np.unique(self.subjects)}")
        print("=" * 80)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_tc = torch.FloatTensor(self.X[idx])      # (T, C)
        X_ct = X_tc.transpose(0, 1)                # (C, T)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        s = int(self.subjects[idx])
        return X_ct, y, s

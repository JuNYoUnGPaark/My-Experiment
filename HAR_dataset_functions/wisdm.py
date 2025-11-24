# datasets_wisdm.py

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class WISDMDataset(Dataset):
    """
    WISDM TXT Loader
    txt format:
        subject, activity, timestamp, x, y, z;
    슬라이딩 윈도우 → (C, T) 형태로 반환.
    """
    def __init__(self, file_path: str, window_size: int = 80, step_size: int = 40):
        super().__init__()

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"[WISDM] Not found: {file_path}")

        self.window_size = window_size
        self.step_size = step_size

        df = self._load_file(file_path)
        self.X, self.y, self.subjects = self._create_windows(df)

        print("=" * 80)
        print("[WISDM] Loaded")
        print(f"  X shape       : {self.X.shape}  (N, T, C)")
        print(f"  y shape       : {self.y.shape}")
        print(f"  unique subjects: {sorted(np.unique(self.subjects))}")
        print("=" * 80)

    # ---------------------------------------------------------
    # RAW TXT → DataFrame
    # ---------------------------------------------------------
    def _load_file(self, file_path: str) -> pd.DataFrame:
        rows = []
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip().replace(";", "")
                if not line:
                    continue

                parts = line.split(",")
                if len(parts) != 6:
                    continue

                subj, act, ts, x, y, z = parts
                rows.append([subj, act, ts, x, y, z])

        df = pd.DataFrame(rows, columns=["subject", "activity", "timestamp", "x", "y", "z"])

        df.replace(["", "NaN", "nan"], np.nan, inplace=True)
        df.dropna(subset=["subject", "x", "y", "z"], inplace=True)

        df["subject"] = df["subject"].astype(int)
        df["x"] = df["x"].astype(float)
        df["y"] = df["y"].astype(float)
        df["z"] = df["z"].astype(float)

        df["activity_id"] = df["activity"].astype("category").cat.codes
        return df

    # ---------------------------------------------------------
    # Sliding Window
    # ---------------------------------------------------------
    def _create_windows(self, df: pd.DataFrame):
        X_list, y_list, s_list = [], [], []

        for subj_id in sorted(df["subject"].unique()):
            df_sub = df[df["subject"] == subj_id].reset_index(drop=True)

            data = df_sub[["x", "y", "z"]].to_numpy(dtype=np.float32)
            labels = df_sub["activity_id"].to_numpy(dtype=np.int64)
            L = len(df_sub)

            start = 0
            while start + self.window_size <= L:
                end = start + self.window_size

                window_x = data[start:end]      # (T, 3)
                window_y = labels[end - 1]      # 마지막 라벨 사용

                X_list.append(window_x)
                y_list.append(window_y)
                s_list.append(subj_id)

                start += self.step_size

        X = np.stack(X_list, axis=0).astype(np.float32)   # (N, T, 3)
        y = np.array(y_list, dtype=np.int64)
        s = np.array(s_list, dtype=np.int64)

        return X, y, s

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_tc = torch.FloatTensor(self.X[idx])       # (T, C)
        X_ct = X_tc.transpose(0, 1)                 # (C, T)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        s = int(self.subjects[idx])
        return X_ct, y, s

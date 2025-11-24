# datasets_mhealth.py

import os
import glob
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


# ============================================================
# 1) MHEALTH 단일 로그 로더
# ============================================================

def _load_single_mhealth_log(path: str, feature_cols: list[str]):
    """
    하나의 mHealth_subjectXX.log 파일을 로드해서 DataFrame으로 반환.
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=feature_cols + ["label"],
    )
    return df


def load_mhealth_dataframe(data_dir: str):
    """
    data_dir 안의 mHealth_subject*.log 전부 읽어서 하나의 DataFrame으로 concat.
    0 (Null class) 제거 + 라벨을 0~11로 shift.
    """
    feature_cols = [
        "acc_chest_x", "acc_chest_y", "acc_chest_z",
        "ecg_1",       "ecg_2",
        "acc_ankle_x", "acc_ankle_y", "acc_ankle_z",
        "gyro_ankle_x","gyro_ankle_y","gyro_ankle_z",
        "mag_ankle_x", "mag_ankle_y","mag_ankle_z",
        "acc_arm_x",   "acc_arm_y",  "acc_arm_z",
        "gyro_arm_x",  "gyro_arm_y", "gyro_arm_z",
        "mag_arm_x",   "mag_arm_y",  "mag_arm_z",
    ]  # 총 23 channels

    log_files = glob.glob(os.path.join(data_dir, "mHealth_subject*.log"))
    if not log_files:
        raise FileNotFoundError(f"[MHEALTHDataset] No mHealth_subject*.log in {data_dir}")

    dfs = [_load_single_mhealth_log(fp, feature_cols) for fp in log_files]
    full_df = pd.concat(dfs, ignore_index=True)

    # Null 클래스 (label == 0) 제거
    full_df = full_df[full_df["label"] != 0].copy()

    # 원래 라벨 1~12 → 0~11로 shift
    full_df["label"] = full_df["label"] - 1

    return full_df, feature_cols


# ============================================================
# 2) 슬라이딩 윈도우 생성
# ============================================================

def create_mhealth_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    step_size: int,
):
    """
    전체 시계열을 window_size 기준으로 슬라이딩 윈도우 생성.
    반환:
        X_np : (N, C, T)
        y_np : (N,)
    """
    data_arr   = df[feature_cols].to_numpy(dtype=np.float32)  # (L, C)
    labels_arr = df["label"].to_numpy(dtype=np.int64)         # (L,)
    L = len(df)

    X_list, y_list = [], []

    start = 0
    while start + window_size <= L:
        end = start + window_size

        window_x = data_arr[start:end]            # (T, C)
        window_y = labels_arr[end - 1]            # 마지막 timestep label 사용
        
        # (T, C) → (C, T)
        window_ct = np.transpose(window_x, (1, 0))

        X_list.append(window_ct)
        y_list.append(int(window_y))

        start += step_size

    if not X_list:
        raise RuntimeError("[MHEALTH] No windows produced. Check window_size/step_size.")

    X_np = np.stack(X_list, axis=0).astype(np.float32)   # (N, C, T)
    y_np = np.array(y_list, dtype=np.int64)              # (N,)

    return X_np, y_np


# ============================================================
# 3) MHEALTH Dataset 클래스
# ============================================================

class MHEALTHDataset(Dataset):
    """
    MHEALTH Dataset - (C, T) 형태로 윈도우 생성
    __getitem__ → (tensor(C, T), y, subject dummy)
    """
    def __init__(self, data_dir: str, window_size: int = 128, step_size: int = 64):
        super().__init__()

        full_df, feature_cols = load_mhealth_dataframe(data_dir)

        X, y = create_mhealth_windows(
            df=full_df,
            feature_cols=feature_cols,
            window_size=window_size,
            step_size=step_size,
        )

        self.X = X                                  # (N, C, T)
        self.y = y                                  # (N,)
        self.subjects = np.zeros(len(self.y), dtype=int)   # dummy subject ID

        # 클래스 이름: 12 classes
        self.label_names = [
            "Standing still", "Sitting and relaxing", "Lying down",
            "Walking", "Climbing stairs", "Waist bends forward",
            "Frontal elevation of arms", "Knees bending", "Cycling",
            "Jogging", "Running", "Jump front & back",
        ]

        print("=" * 80)
        print("[MHEALTHDataset] Loaded")
        print(f"  X shape : {self.X.shape}  (N, C, T)")
        print(f"  y shape : {self.y.shape}  (N,)")
        print(f"  Classes : {len(self.label_names)}")
        print("=" * 80)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_ct = torch.FloatTensor(self.X[idx])      # (C, T)
        y    = torch.tensor(self.y[idx], dtype=torch.long)
        s    = int(self.subjects[idx])             # dummy
        return X_ct, y, s

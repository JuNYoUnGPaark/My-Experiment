# datasets_pamap2.py

import os
import re
import glob
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. PAMAP2 관련 상수/매핑
# ============================================================

# 사용할 피처들 (orientation, heart rate 등은 제외)
PAMAP2_FEATURE_COLS = [
    # hand
    "handAcc16_1", "handAcc16_2", "handAcc16_3",
    "handAcc6_1",  "handAcc6_2",  "handAcc6_3",
    "handGyro1",   "handGyro2",   "handGyro3",
    "handMagne1",  "handMagne2",  "handMagne3",
    # chest
    "chestAcc16_1", "chestAcc16_2", "chestAcc16_3",
    "chestAcc6_1",  "chestAcc6_2",  "chestAcc6_3",
    "chestGyro1",   "chestGyro2",   "chestGyro3",
    "chestMagne1",  "chestMagne2",  "chestMagne3",
    # ankle
    "ankleAcc16_1", "ankleAcc16_2", "ankleAcc16_3",
    "ankleAcc6_1",  "ankleAcc6_2",  "ankleAcc6_3",
    "ankleGyro1",   "ankleGyro2",   "ankleGyro3",
    "ankleMagne1",  "ankleMagne2",  "ankleMagne3",
]

# PAMAP2 원래 activityID 중 사용할 12개 클래스
ORDERED_ACTIVITY_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]

# 원본 activityID -> new index (0..11)
PAMAP2_OLD2NEW = {
    1: 0,   # Lying
    2: 1,   # Sitting
    3: 2,   # Standing
    4: 3,   # Walking
    5: 4,   # Running
    6: 5,   # Cycling
    7: 6,   # Nordic walking
    12: 7,  # Ascending stairs
    13: 8,  # Descending stairs
    16: 9,  # Vacuum cleaning
    17: 10, # Ironing
    24: 11, # Rope jumping
}

# new index -> 사람이 읽을 수 있는 이름
PAMAP2_LABEL_NAMES = [
    "Lying",              # 0 -> orig 1
    "Sitting",            # 1 -> orig 2
    "Standing",           # 2 -> orig 3
    "Walking",            # 3 -> orig 4
    "Running",            # 4 -> orig 5
    "Cycling",            # 5 -> orig 6
    "Nordic walking",     # 6 -> orig 7
    "Ascending stairs",   # 7 -> orig 12
    "Descending stairs",  # 8 -> orig 13
    "Vacuum cleaning",    # 9 -> orig 16
    "Ironing",            # 10 -> orig 17
    "Rope jumping",       # 11 -> orig 24
]


# ============================================================
# 2. 슬라이딩 윈도우 생성 함수
# ============================================================

def create_pamap2_windows(df: pd.DataFrame,
                          window_size: int,
                          step_size: int):
    """
    subject별로 timestamp 순서대로 전체 시계열을 따라가며 슬라이딩 윈도우 생성.
    한 윈도우의 라벨은 마지막 프레임의 activityID(원본) 기준으로 설정하고,
    0 (Null/기타) 이거나 우리가 사용하는 12개 클래스에 속하지 않는 경우는 버린다.

    Returns:
        X:        (N, C, T) float32
        y:        (N,) int64, new index 0..11
        subj_ids: (N,) int64
        label_names: list[str]  길이 12
    """
    X_list = []
    y_list = []
    subj_list = []

    for subj_id, g in df.groupby("subject_id"):
        # 시간순 정렬
        if "timestamp" in g.columns:
            g = g.sort_values("timestamp")
        else:
            g = g.sort_index()

        data_arr = g[PAMAP2_FEATURE_COLS].to_numpy(dtype=np.float32)  # (L, C)
        label_arr = g["activityID"].to_numpy(dtype=np.int64)          # (L,)
        L = data_arr.shape[0]

        start = 0
        while start + window_size <= L:
            end = start + window_size

            last_label_orig = int(label_arr[end - 1])

            # 0 = "other / null" → 버림
            if last_label_orig == 0:
                start += step_size
                continue

            # 우리가 쓰는 12개 클래스가 아니면 버림
            if last_label_orig not in PAMAP2_OLD2NEW:
                start += step_size
                continue

            # 윈도우 추출 (C, T) 형태로
            window_ct = data_arr[start:end].T  # (T, C) → (C, T)

            X_list.append(window_ct)
            y_list.append(PAMAP2_OLD2NEW[last_label_orig])
            subj_list.append(int(subj_id))

            start += step_size

    if len(X_list) == 0:
        raise RuntimeError(
            "No valid PAMAP2 windows created. "
            "Check window_size / step_size / activityID filtering."
        )

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, C, T)
    y = np.asarray(y_list, dtype=np.int64)           # (N,)
    subj_ids = np.asarray(subj_list, dtype=np.int64)

    return X, y, subj_ids, PAMAP2_LABEL_NAMES


# ============================================================
# 3. PAMAP2 Dataset 클래스
# ============================================================

class PAMAP2Dataset(Dataset):
    """
    PAMAP2 Dataset (windowed)

    - data_dir 이하의 *.csv 파일들을 모두 읽어서 concat
    - subject_id, activityID, timestamp, 센서 피처들을 이용
    - subject별로 NaN 보간 + 표준화( StandardScaler )
    - 슬라이딩 윈도우로 (N, C, T), y, subj_ids 생성

    __getitem__:
        X: (C, T) float32
        y: scalar int64
        subject_id: scalar int
    """
    def __init__(self,
                 data_dir: str,
                 window_size: int = 500,
                 step_size: int = 250):
        super().__init__()

        # 1) CSV 전부 읽어서 하나의 df로 합치기
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if len(csv_files) == 0:
            raise RuntimeError(f"[PAMAP2Dataset] No CSV files found under {data_dir}")

        dfs = []
        for fpath in sorted(csv_files):
            df_i = pd.read_csv(fpath)

            # subject_id 칼럼이 없으면 파일명에서 유추
            if "subject_id" not in df_i.columns:
                m = re.findall(r"\d+", os.path.basename(fpath))
                subj_guess = int(m[0]) if len(m) > 0 else 0
                df_i["subject_id"] = subj_guess

            dfs.append(df_i)

        df = pd.concat(dfs, ignore_index=True)

        # activityID / subject_id / timestamp 타입 정리
        if "activityID" not in df.columns:
            raise RuntimeError("[PAMAP2Dataset] 'activityID' column not found in CSVs.")

        df = df.dropna(subset=["activityID"])
        df["activityID"] = df["activityID"].astype(np.int64)
        df["subject_id"] = df["subject_id"].astype(np.int64)

        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

        # ---------------------------
        # (1) NaN 처리 (subject별 보간)
        # ---------------------------
        def _fill_subject_group(g):
            # timestamp 있으면 시간 순 정렬
            if "timestamp" in g.columns:
                g = g.sort_values("timestamp")
            else:
                g = g.sort_index()

            g[PAMAP2_FEATURE_COLS] = (
                g[PAMAP2_FEATURE_COLS]
                .interpolate(method="linear", limit_direction="both", axis=0)
                .ffill()
                .bfill()
            )
            return g

        df = df.groupby("subject_id", group_keys=False).apply(_fill_subject_group)

        # safety net: 혹시라도 남은 NaN은 0으로
        df[PAMAP2_FEATURE_COLS] = df[PAMAP2_FEATURE_COLS].fillna(0.0)

        # ---------------------------
        # (2) 스케일 표준화 (전체 기준)
        # ---------------------------
        scaler = StandardScaler()
        df[PAMAP2_FEATURE_COLS] = scaler.fit_transform(df[PAMAP2_FEATURE_COLS])

        # ---------------------------
        # (3) 슬라이딩 윈도우 생성
        # ---------------------------
        X, y, subj_ids, label_names = create_pamap2_windows(
            df=df,
            window_size=window_size,
            step_size=step_size,
        )

        self.X = X                  # (N, C, T)
        self.y = y                  # (N,)
        self.subject_ids = subj_ids # (N,)
        self.label_names = label_names

        print("=" * 80)
        print("[PAMAP2Dataset] Loaded")
        print(f"  data_dir    : {data_dir}")
        print(f"  X shape     : {self.X.shape}  (N, C, T)")
        print(f"  y shape     : {self.y.shape}   (N,)")
        print(f"  subjects    : {np.unique(self.subject_ids)}")
        print(f"  window_size : {window_size}, step_size: {step_size}")
        print("=" * 80)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()         # (C, T)
        y = torch.tensor(self.y[idx], dtype=torch.long)   # scalar
        s = int(self.subject_ids[idx])                    # scalar
        return x, y, s

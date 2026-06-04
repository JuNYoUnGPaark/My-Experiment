MOTIONSENSE_ACTIVITIES = ['dws', 'jog', 'sit', 'std', 'ups', 'wlk']

class MotionSenseDataset(Dataset):
    def __init__(
        self,
        root_dir,
        window_size=128,
        step_size=64,
        normalize=True,
        target_subjects=None,
        scaler=None
    ):
        self.root_dir = Path(root_dir)
        self.window_size = window_size
        self.step_size = step_size
        self.normalize = normalize

        self.data_dir = self.root_dir / "A_DeviceMotion_data"

        df_all = self._load_all_data()

        if target_subjects is not None:
            df_all = df_all[df_all['subject_id'].isin(target_subjects)].copy()
            print(f"Dataset initialized with subjects: {target_subjects}")
            print(f"Total rows after filtering: {len(df_all)}")

        activities = MOTIONSENSE_ACTIVITIES  # 고정 순서
        self.label2idx = {label: i for i, label in enumerate(activities)}
        self.idx2label = {i: label for label, i in self.label2idx.items()}
        df_all["label_idx"] = df_all["activity"].map(self.label2idx)

        feat_cols = [
            "userAcceleration.x", "userAcceleration.y", "userAcceleration.z",
            "rotationRate.x", "rotationRate.y", "rotationRate.z"
        ]
        feats = df_all[feat_cols].values.astype(np.float32)

        if self.normalize:
            if scaler is None:
                self.scaler = StandardScaler()
                feats = self.scaler.fit_transform(feats)
            else:
                self.scaler = scaler
                feats = self.scaler.transform(feats)
        else:
            self.scaler = None

        df_all[feat_cols] = feats

        X_list, y_list = [], []
        for _, g in df_all.groupby(["subject_id", "activity", "trial_id"]):
            g = g.sort_values("timestamp_idx").reset_index(drop=True)

            data = g[feat_cols].values
            labels = g["label_idx"].values
            n = len(g)
            if n < window_size:
                continue

            for start in range(0, n - window_size + 1, step_size):
                end = start + window_size
                w_data = data[start:end]
                w_labels = labels[start:end]
                majority_label = np.bincount(w_labels).argmax()

                X_list.append(w_data.astype(np.float32))  
                y_list.append(int(majority_label))

        self.X = np.stack(X_list) if len(X_list) > 0 else np.zeros((0, window_size, 6), dtype=np.float32)
        self.y = np.array(y_list, dtype=np.int64)

        print(f"[MotionSenseDataset] windows: {len(self.X)}, classes: {len(self.label2idx)}")
        print(f"Classes map: {self.label2idx}")

    def _load_all_data(self):
        all_dfs = []
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        for folder in os.listdir(self.data_dir):
            folder_path = self.data_dir / folder
            if not folder_path.is_dir():
                continue

            parts = folder.split('_')
            activity_label = parts[0]
            subject_id = parts[1]

            for csv_file in os.listdir(folder_path):
                if not csv_file.endswith(".csv"):
                    continue

                file_path = folder_path / csv_file
                df = pd.read_csv(file_path)

                if "Unnamed: 0" in df.columns:
                    df = df.rename(columns={"Unnamed: 0": "timestamp_idx"})
                else:
                    df["timestamp_idx"] = range(len(df))

                df["activity"] = activity_label
                df["subject_id"] = int(subject_id)
                df["trial_id"] = folder
                all_dfs.append(df)

        if len(all_dfs) == 0:
            raise RuntimeError(f"No MotionSense csv files found under: {self.data_dir}")

        return pd.concat(all_dfs, ignore_index=True)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.long)

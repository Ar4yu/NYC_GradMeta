import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    """
    Returns ONE sample: the full sequence from a CSV.

    - X: [T, F] (all numeric columns except the target + optional date)
    - y: [T]    (target column)
    """
    def __init__(self, csv_file: str, target_name: str, date_col: str = "date", keep_date: bool = False):
        df = pd.read_csv(csv_file)

        if target_name not in df.columns:
            raise ValueError(f"Target column '{target_name}' not found in {csv_file}. Columns={df.columns.tolist()}")

        # Keep date for debugging/joins if you want, but model should not ingest it as numeric feature.
        self.dates = None
        if date_col in df.columns:
            self.dates = pd.to_datetime(df[date_col])
            if not keep_date:
                df = df.drop(columns=[date_col])

        y = df[target_name].to_numpy(dtype=np.float32)
        X = df.drop(columns=[target_name]).to_numpy(dtype=np.float32)

        # store as tensors
        self.X = torch.from_numpy(X)         # [T, F]
        self.y = torch.from_numpy(y)         # [T]

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.X, self.y
import os
import numpy as np
import pandas as pd
from config import ALL_FEATURES

def get_normalization_params(data):
    mean = data.mean(axis=0)
    std = data.std(axis=0) + 1e-5
    return mean, std

def load_seed_data(data_path, context_len, norm_params=None):
    df = pd.read_csv(data_path)
    df = df.sort_values('Time').reset_index(drop=True)
    data = df[ALL_FEATURES].values.astype(np.float32)

    if norm_params:
        mean = np.array(norm_params['mean'])
        std = np.array(norm_params['std'])
    else:
        mean, std = get_normalization_params(data)

    seed_data_raw = data[:context_len]
    seed_data = (seed_data_raw - mean) / std
    return seed_data, mean, std

def load_all_locations(data_dir=None):
    import glob
    if data_dir is None:
        from config import _BASE_DIR
        data_dir = os.path.join(_BASE_DIR, "data")
    csv_files = sorted(glob.glob(os.path.join(data_dir, "Location*.csv")))
    all_dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df = df.sort_values('Time').reset_index(drop=True)
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df

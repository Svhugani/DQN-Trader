import os
import pandas as pd


def get_raw_data(filename: str):
    df = pd.read_csv(filename)
    return df.values


def try_make_dir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

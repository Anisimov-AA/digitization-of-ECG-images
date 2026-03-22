import cv2
import numpy as np
import pandas as pd
import yaml
import os


def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def read_image(sample_id, kaggle_dir):
    path = f"{kaggle_dir}/test/{sample_id}.png"
    return cv2.imread(path, cv2.IMREAD_COLOR_RGB)


def read_sampling_length(sample_id, valid_df):
    d = valid_df[
        (valid_df["id"] == sample_id) & (valid_df["lead"] == "II")
    ].iloc[0]
    return d.number_of_rows


def load_test_metadata(kaggle_dir):
    df = pd.read_csv(f"{kaggle_dir}/test.csv")
    df["id"] = df["id"].astype(str)
    return df
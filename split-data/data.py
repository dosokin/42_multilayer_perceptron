import pandas as pd
import numpy as np


def add_columns_to_raw_data_frame(df):
    df.columns = [
        "id",
        "diagnostic",

        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave_points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",

        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave_points_se",
        "symmetry_se",
        "fractal_dimension_se",

        "radius_largest",
        "texture_largest",
        "perimeter_largest",
        "area_largest",
        "smoothness_largest",
        "compactness_largest",
        "concavity_largest",
        "concave_points_largest",
        "symmetry_largest",
        "fractal_dimension_largest",
    ]

    return df


def clear_data(df):
    df = df.drop_duplicates()

    df = df.dropna()

    df = df[(df != 0.0).all(axis=1)]

    return df


def normalize_data(df):

    df = df.drop(['id'], axis=1)

    df['diagnostic'] = df['diagnostic'].str.replace("B", "0")
    df['diagnostic'] = df['diagnostic'].str.replace("M", "1")
    df['diagnostic'] = pd.to_numeric(df['diagnostic'])

    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
        df[col] = df[col] / df[col].max()

    return df


def filter_features(df, features_count):

    diagnostic_corr = df.corr()['diagnostic'].sort_values(ascending=False)

    features_list = diagnostic_corr.index.values.tolist()

    features_count = min(max(features_count, 1), features_count - 1)

    return df[features_list[:features_count + 1]]


def split_data_frame(df, training_allocation, seed):

    if seed:
        np.random.seed(0)

    mask = np.random.rand(len(df)) < (training_allocation / 100.0)

    train_set = df[mask]
    val_set = df[~mask]

    return train_set, val_set

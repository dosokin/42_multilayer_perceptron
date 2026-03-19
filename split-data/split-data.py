import pandas as pd
from pathlib import Path

from args import parse_args
from data import (
    add_columns_to_raw_data_frame,
    normalize_data, filter_features,
    clear_data, split_data_frame)
from utils import df_to_csv


def split_data(filename="../data.csv", training_allocation=80,
               output_folder="./data", features_count=26, seed=False):

    file_path = Path(filename)
    if not file_path.exists():
        print(f"Specified file doesnt exists: {file_path.absolute()}")
        return

    try:
        df = pd.read_csv(file_path.absolute())
    except Exception as e:
        print(f"Error reading the data source file: {e}")
        return

    df = add_columns_to_raw_data_frame(df)
    df = normalize_data(df)
    df = filter_features(df, features_count)
    df = clear_data(df)

    train_set, val_set = split_data_frame(df, training_allocation, seed)

    try:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        df_to_csv(train_set, output_folder / "train.csv")
        df_to_csv(val_set, output_folder / "validation.csv")
    except Exception as e:
        print(e)


if __name__ == "__main__":

    args = parse_args()

    split_data(
        filename=args.filename,
        training_allocation=args.split,
        output_folder=args.output,
        features_count=args.features,
        seed=args.seed
    )

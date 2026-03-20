import pandas as pd
from pathlib import Path

from multilayer_perceptron.split.args import parse_args
from multilayer_perceptron.split.data import (
    add_columns_to_raw_data_frame,
    normalize_data,
    clean_data, split_data_frame)
from multilayer_perceptron.split.utils import df_to_csv


def format_data(df):
    df = add_columns_to_raw_data_frame(df)
    df = normalize_data(df)
    df = clean_data(df)

    return df


def split_data(filename="../data.csv", training_allocation=80,
               output_folder="../data", seed=False):

    file_path = Path(filename)
    if not file_path.exists():
        print(f"Specified file doesnt exists: {file_path.absolute()}")
        return

    try:
        df = pd.read_csv(file_path.absolute())
    except Exception as e:
        print(f"Error reading the data source file: {e}")
        return

    df = format_data(df)

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
        seed=args.seed
    )

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        help="Seed to always get the same split output",
        action="store_true"
    )

    parser.add_argument(
        "-f",
        "--filename",
        help="Input data path. If not specified ./data.csv by default",
        type=str,
        default="data.csv"
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output folder. If not specified ./data by default",
        type=str,
        default="data"
    )

    parser.add_argument(
        "-s",
        "--split",
        nargs=3,
        type=int,
        help="Specify the part of the dataset for each set in percent.\n\
Usage: --split a\na => training set allocation in percentage\nBy default 80",
        default=80,
    )

    return parser.parse_args()

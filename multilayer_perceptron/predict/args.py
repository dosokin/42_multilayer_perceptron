import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "datapath",
        help="path to data to predict",
        type=str
    )

    parser.add_argument(
        "-m",
        "--model",
        help="path to model to load",
        type=str,
        default="models/model.json"
    )

    return parser.parse_args()
import argparse

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--architecture",
        help="Specify the network architecture, \
each argument represents the number of neurons of the layer",
        type=int,
        nargs="*",
        default=[16, 8, 2]
    )

    parser.add_argument(
        "-l",
        "--learning-rate",
        help="Set the learning rate",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "-i",
        "--input-folder",
        help="Folder storing splitted data",
        type=str,
        default="data"
    )

    parser.add_argument(
        "-o",
        "--output-file",
        help="Model saving filepath",
        default="model.json",
        type=str,
    )

    parser.add_argument(
        "--max-epoch",
        help="Set the max epoch",
        default=10000,
        type=int,
    )

    parser.add_argument(
        "--patience",
        help="Set the number of epoch without model improvements before early stop",
        default=20,
        type=int,
    )

    parser.add_argument(
        "--warming",
        help="Set the number of epochs without improvements required",
        default=100,
        type=int,
    )

    return parser.parse_args()


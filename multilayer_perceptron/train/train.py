import pandas as pd

from pathlib import Path

from multilayer_perceptron.core.model import ModelPerformance, Model
from multilayer_perceptron.train.args import parse_args
from multilayer_perceptron.core.neural_network import NeuralNetwork
from multilayer_perceptron.train.epoch import epoch
from multilayer_perceptron.train.graphs import display_graphs


def train(architecture=(16, 8, 2), lr=0.01,
          in_folder="data", out_file="model.json",
          max_epoch=10000, patience=20, warming=100):

    try:
        data_folder = Path(in_folder)
        train_set = pd.read_csv(data_folder / "train.csv").iloc[:,1:].to_numpy()
        val_set = pd.read_csv(data_folder / "validation.csv").iloc[:,1:].to_numpy()
    except Exception as e:
        print(f"Error reading data files: {e}")
        return

    try:
        nn = NeuralNetwork(
            train_set.shape[1] - 1,
            *architecture
        )
    except Exception as e:
        print(f"Error initialising NeuralNetwork: {e}")
        return

    best_model = ModelPerformance(
        loss=None,
        model=None
    )

    evolution_accuracy = []
    evolution_train_loss = []
    evolution_val_loss = []

    p = 0
    epoch_count = 0

    for x in range(max_epoch):
        epoch_r = epoch(nn, train_set, val_set, lr)

        epoch_count += 1

        evolution_accuracy.append(epoch_r.val_accuracy)
        evolution_train_loss.append(epoch_r.train_loss)
        evolution_val_loss.append(epoch_r.val_loss)

        if (best_model.loss is None or
                epoch_r.val_loss < best_model.loss):
            best_model.loss = epoch_r.val_loss
            best_model.model = nn.get_model()
            p = 0

        elif x > warming and epoch_r.val_loss > best_model.loss + 0.01:
            p += 1

        if p > patience:
            break

        print(f"epoch {x} - loss: {epoch_r.train_loss:.4f} - val_loss: {epoch_r.val_loss:.4f}")

    try:
        best_model.model.save(out_file)
    except Exception as e:
        print(e)

    display_graphs(evolution_accuracy, evolution_train_loss, evolution_val_loss, epoch_count)


if __name__ == "__main__":
    args = parse_args()

    train(
        architecture=args.architecture,
        lr=args.learning_rate,
        in_folder=args.input_folder,
        out_file=args.output_file,
        max_epoch=args.max_epoch,
        patience=args.patience,
        warming=args.warming
    )
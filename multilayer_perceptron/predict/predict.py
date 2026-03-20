import json
import numpy as np

from pathlib import Path

import pandas as pd

from multilayer_perceptron.predict.args import parse_args
from multilayer_perceptron.core.neural_network import Model, LayerParameters, NeuralNetwork
from multilayer_perceptron.core.maths import binary_cross_entropy_error


def file_to_model(filepath):
    input_file = Path(filepath)

    if not input_file.exists():
        raise Exception(f"{input_file.absolute()}: Model file doesnt exists")

    raw_content = input_file.read_text()

    if not raw_content:
        raise Exception(f"{input_file.absolute()}: model couldnt be read")

    raw_model = json.loads(raw_content)

    layers = [
        LayerParameters(
            weights=np.array(x["weights"]),
            biases=np.array(x["biases"])
        ) for x in raw_model["layers"]
    ]

    model = Model(
        layers
    )

    return model

def is_correct(expected, actual):
    return (actual[0] > actual[1]) != (expected[1] == 1)

def iteration(nn, sample):
    expected_result = np.array([[sample[0]], [1 - sample[0]]])
    model_input = np.reshape(sample[1:], [len(sample) - 1, 1])

    model_output = nn.init_forward_pass(model_input)

    loss = binary_cross_entropy_error(model_output, expected_result)

    return is_correct(expected_result.flatten(), model_output.flatten()), loss

def validation_epoch(nn, prediction_set):

    cumulate_loss = 0
    correct_output = 0
    for sample in prediction_set:
        correct, loss = iteration(nn, sample)
        correct_output += int(correct)
        cumulate_loss += loss

    final_accuracy = correct_output / len(prediction_set)
    final_loss = cumulate_loss / len(prediction_set)

    return final_accuracy, final_loss


def predict(data_file, model_file="models/model.json"):

    try:
        model = file_to_model(model_file)
    except Exception as e:
        print(f"Error loading the model from {model_file}: {e}")
        return

    try:
        prediction_set = pd.read_csv(data_file).iloc[:,1:].to_numpy()
    except Exception as e:
        print(f"Error loading the data from {data_file}: {e}")
        return

    nn = NeuralNetwork(
        model=model
    )

    accuracy, loss = validation_epoch(nn, prediction_set)

    print(f"ACCURACY: {accuracy} - LOSS: {loss}")


if __name__ == "__main__":
    args = parse_args()

    predict(
        data_file=args.datapath,
        model_file=args.model
    )
import numpy as np
from dataclasses import dataclass

from multilayer_perceptron.core.maths import binary_cross_entropy_error

@dataclass
class EpochResult:
    train_loss: float
    val_loss: float
    val_accuracy: float

def training_iteration(nn, sample, lr):

    expected_result = np.array([[sample[0]], [1 - sample[0]]])
    model_input = np.reshape(sample[1:], [len(sample) - 1, 1])

    model_output = nn.init_forward_pass(model_input)

    loss = binary_cross_entropy_error(model_output, expected_result)

    nn.init_back_propagation(expected_result, lr)

    return loss

def is_correct(expected, actual):
    return (actual[0] > actual[1]) != (expected[1] == 1)

def validation_iteration(nn, sample):
    expected_result = np.array([[sample[0]], [1 - sample[0]]])
    model_input = np.reshape(sample[1:], [len(sample) - 1, 1])

    model_output = nn.init_forward_pass(model_input)

    loss = binary_cross_entropy_error(model_output, expected_result)

    return is_correct(expected_result.flatten(), model_output.flatten()), loss


def validation_epoch(nn, test_set):

    cumulate_loss = 0
    correct_output = 0
    for sample in test_set:
        correct, loss = validation_iteration(nn, sample)
        correct_output += int(correct)
        cumulate_loss += loss

    final_accuracy = correct_output / len(test_set)
    final_loss = cumulate_loss / len(test_set)

    return final_accuracy, final_loss

def training_epoch(nn, train_set, lr):

    cumulate_loss = 0
    for sample in train_set:
        loss = training_iteration(nn, sample, lr)
        cumulate_loss += loss

    final_loss = cumulate_loss / len(train_set)

    return final_loss

def epoch(nn, train_set, val_set, lr):

    train_loss = training_epoch(nn, train_set, lr)

    val_accuracy, val_loss = validation_epoch(nn, val_set)

    return EpochResult(train_loss, val_loss, val_accuracy)
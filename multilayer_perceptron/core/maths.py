import numpy as np

def softmax(x):

  soft_x = x / x.sum()

  return soft_x


def binary_cross_entropy_error(predictions, expected):

  def f(p, y):
    return y * np.log(p) + (1 - y) * np.log(1 - p)

  loss = f(predictions, expected).sum() * -1 / len(predictions)

  return loss


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
  return sigmoid(x) * (1 - sigmoid(x))


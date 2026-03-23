from dataclasses import dataclass

import numpy as np

from multilayer_perceptron.core.maths import softmax, sigmoid, d_sigmoid

@dataclass
class LayerParameters:
    weights: np.ndarray
    biases: np.ndarray


class Layer:

  def __init__(self,
               parameters: LayerParameters,
               previous_layer=None):

    self.W = parameters.weights
    self.B = parameters.biases

    self.Z = None
    self.output = None
    self.C = None

    self.previous_layer = previous_layer
    if self.previous_layer:
      self.previous_layer.next_layer = self

    self.next_layer = None
    self.input = None
    self.delta = None

  def forward_pass(self, a, result=False):

    self.input = a

    self.Z = self.W @ self.input + self.B

    self.output = sigmoid(self.Z)

    if self.next_layer:
      return self.next_layer.forward_pass(self.output, result=result)

    return softmax(self.output)

  def set_delta(self, expected_y=None):

    if self.next_layer is None:
      self.delta = self.output - expected_y

    else:
      self.delta = np.transpose(self.next_layer.W) @ self.next_layer.delta * d_sigmoid(self.Z)

    if self.previous_layer:
      self.previous_layer.set_delta()

  def set_c(self):
    self.C = self.delta @ np.transpose(self.input)
    if self.previous_layer:
      self.previous_layer.set_c()

  def adjust_weights(self, lr):
    self.W = self.W - (lr * self.C)
    if self.previous_layer:
      self.previous_layer.adjust_weights(lr)

  def adjust_biases(self, lr):
      self.B = self.B - (lr * self.delta)
      if self.previous_layer:
          self.previous_layer.adjust_biases(lr)

  def get_parameters(self):
        return LayerParameters(
            self.W,
            self.B
        )
import numpy as np

from multilayer_perceptron.core.layer import Layer, LayerParameters
from multilayer_perceptron.core.model import Model


class NeuralNetwork:

    def __init__(self, *args, model=None):
        if model:
            self.load_model(model)
            return

        if len(args) <= 1:
            raise Exception("The neural network should have at least one layer")

        self.architecture = [int(x) for x in args[1:]]

        self.layers = []

        current_layer = None
        input_count = int(args[0])
        for neurons_count in self.architecture:
            current_layer = Layer (
                LayerParameters(
                    weights=np.random.rand(neurons_count, input_count),
                    biases=np.reshape(np.zeros(neurons_count), [neurons_count, 1])
                ),
                previous_layer=current_layer,
            )
            input_count = neurons_count
            self.layers.append(current_layer)

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

    def load_model(self, model: Model):

        self.layers = []
        current_layer = None
        for layer in model.layers:
            current_layer = Layer(
                parameters=layer,
                previous_layer=current_layer
            )
            self.layers.append(current_layer)

        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]

    def init_forward_pass(self, x):
        return self.input_layer.forward_pass(x)

    def init_back_propagation(self, expected_y, lr):
        self.output_layer.set_delta(expected_y=expected_y)
        self.output_layer.set_c()
        self.output_layer.adjust_weights(lr)
        self.output_layer.adjust_biases(lr)

    def get_model(self):

        layers_params = []
        for l in self.layers:
            layers_params.append(l.get_parameters())

        return Model(
            layers_params
        )

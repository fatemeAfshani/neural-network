import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


X, y = spiral_data(100, 3)


class LayerDense:
    def __init__(self, n_inputs, n_outputs):
        self.outputs = None
        self.weights = np.random.randn(n_inputs, n_outputs)
        self.biases = np.zeros((1, n_outputs))
        self.activation = None
        self.dropout = None
        self.loss = None
        self.accuracy = None
        self.optimizer = None
        self.learning_rate = None

    def forward(self, inputs):
        self.outputs = np.dot(inputs, self.weights) + self.biases


class Activation:
    def __init__(self):
        self.output = None

    def forward(self, inputData):
        self.output = np.maximum(0, inputData)


layer1 = LayerDense(2, 5)
activation1 = Activation()

layer1.forward(X)
activation1.forward(layer1.outputs)
print(activation1.output)

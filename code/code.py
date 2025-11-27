import numpy as np

inputs = [
    [1.2, 5.1, 2.1, 3],
    [3.2, 2.1, 8, 1],
    [2.2, 1.1, -1, 3],
    ]


class LayerDense:
    def __init__(self, n_inputs, n_outputs):
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


layer1 = LayerDense(4, 5)
layer2 = LayerDense(5, 2)

layer1.forward(inputs)
layer2.forward(layer1.outputs)

print(layer2.outputs)

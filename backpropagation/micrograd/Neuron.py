from Micrograd import Value
import numpy as np

class Neuron:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.weights = [Value(np.random.randn()) for _ in range(input_dim)]
        self.bias = Value(0.0)

    def forward(self, inputs):
        out = self.bias
        for x, w in zip(inputs, self.weights):
            out = out + x * w
        return out

    __call__ = forward

    def parameters(self):
        out = self.weights.copy()
        out.append(self.bias)
        return out
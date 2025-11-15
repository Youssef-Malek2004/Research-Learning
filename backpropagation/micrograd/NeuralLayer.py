from Neuron import Neuron

class NeuralLayer:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

    def forward(self, inputs):
        out = []
        for neuron in self.neurons:
            out.append(neuron.forward(inputs))
        return out

    __call__ = forward

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params += neuron.parameters()
        return params

class ReLULayer:
    def __call__(self, values):
        return [v.relu() for v in values]


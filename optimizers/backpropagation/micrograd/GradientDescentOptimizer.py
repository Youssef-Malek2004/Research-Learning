class GradientDescentOptimizer:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self._lr = lr

    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0.0

    def step(self):
        for param in self.parameters:
            param.data -= self._lr * param.grad
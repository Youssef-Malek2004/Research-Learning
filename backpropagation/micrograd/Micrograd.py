import math

class Value:
    def __init__(self, data, children=(), _op=''):
        self.data = float(data)
        self._prev = set(children)
        self._op = _op

        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data:{self.data}, grad:{self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, children=(self, other), _op='+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, children=(self, other), _op='-')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += -1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, children=(self, other), _op='*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float))

        out = Value(self.data ** exponent, children=(self,), _op=f'**{exponent}')

        def _backward():
            self.grad += (exponent * (self.data ** (exponent - 1))) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0, children=(self,), _op='relu')

        def _backward():
            self.grad += 1 * out.grad if self.data > 0 else 0

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1.0

        for v in reversed(topo):
            v._backward()

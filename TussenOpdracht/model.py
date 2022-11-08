import math
from collections import Counter


# noinspection DuplicatedCode
class Perceptron:
    def __init__(self, dim):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for _ in range(dim)]
        self.predictions = []
        self.check = 0

    def __repr__(self):
        text = f'Perceptron(dim={self.dim})\n' \
               f'- bias = {self.bias}\n- weights = {self.weights}'
        return text

    def predict(self, xs):
        self.predictions = [1 if self.bias + sum(self.weights[i] * point[i] for i in range(self.dim)) > 0 else -1 if
                            self.bias + sum(self.weights[i] * point[i] for i in range(self.dim)) < 0 else 0 for point
                            in xs]
        return self.predictions

    def partial_fit(self, xs, ys):
        i = 0
        for x, y in zip(xs, ys):
            self.predict(xs)
            self.bias = self.bias - (self.predictions[i] - y)
            self.weights = [self.weights[j] - (self.predictions[i] - y) * x[j] for j in range(self.dim)]
            i += 1

    def fit(self, xs, ys, epochs=0):
        if not epochs == 0:
            for i in range(epochs):
                self.partial_fit(xs, ys)
        else:
            current_weight = self.weights
            current_bias = self.bias
            self.partial_fit(xs, ys)
            if current_weight == self.weights and current_bias == self.bias:
                self.check += 1
            else:
                self.check = 0

            if not self.check >= len(xs):
                self.predict(xs)
                self.fit(xs, ys)
            else:
                return


# noinspection DuplicatedCode
class LinearRegression:
    def __init__(self, dim):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for _ in range(dim)]
        self.predictions = []
        self.check = 0

    def __repr__(self):
        text = f'LinearRegression(dim={self.dim})\n' \
               f'- bias = {self.bias}\n- weights = {self.weights}'
        return text

    def predict(self, xs):
        self.predictions = [self.bias + sum(self.weights[i] * point[i] for i in range(self.dim)) for point
                            in xs]
        return self.predictions

    def partial_fit(self, xs, ys, *, alpha=0.01):
        i = 0
        for x, y in zip(xs, ys):
            self.predict(xs)
            self.bias = self.bias - alpha * (self.predictions[i] - y)
            self.weights = [self.weights[j] - alpha * (self.predictions[i] - y) * x[j] for j in range(self.dim)]
            i += 1

    def fit(self, xs, ys, epochs=500, alpha=0.01):
        for i in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)


# Activation functions
def linear(a):
    """
    identity function
    :param a:
    :return a:
    """
    return a


def sign(a):
    """
    signum function
    :param a:
    :return 1, -1 or 0, depending on wether a is positive, negative or 0:
    """
    if a > 0:
        a = 1
    elif a < 0:
        a = -1
    return a


def tanh(a):
    return (math.e ** a - math.e ** - a) / (math.e ** a + math.e ** - a)


# Loss functions
def mean_squared_error(yhat, y):
    return (yhat - y) ** 2


def mean_absolute_error(yhat, y):
    return abs(yhat - y)


def hinge(yhat, y):
    return max(1 - yhat * y, 0)  # max(1−𝑦̂ ⋅𝑦,0)


# Derivative function calculator
def derivative(function, delta=0.01):
    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)

    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    return wrapper_derivative


class Neuron:
    def __init__(self, dim=2, activation=linear, loss=mean_squared_error):
        self.dim = dim
        self.activation = activation
        self.loss = loss
        self.bias = 0
        self.weights = [0 for _ in range(self.dim)]

    def __repr__(self):
        text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
        return text

    def predict(self, xs):
        prediction = []
        for point in xs:
            value = self.bias + sum(self.weights[i] * point[i] for i in range(self.dim))
            prediction.append(self.activation(value))
        return prediction

    def partial_fit(self, xs, ys, *, alpha=0.01):
        # Get predictions
        predictions = self.predict(xs)
        # For every data point adjust bias and weights
        for x, y, yhat in zip(xs, ys, predictions):
            # b <- b - alpha * derivative(loss) * derivative(activation)
            self.bias -= alpha * derivative(self.loss)(yhat, y) * derivative(self.activation)(yhat)
            # wi <- wi - alpha * derivative(loss) * derivative(activation) * xi
            self.weights = [self.weights[j] - alpha * derivative(self.loss)(yhat, y) *
                            derivative(self.activation)(yhat) * x[j] for j in range(self.dim)]  # type: list[int]

    def fit(self, xs, ys, epochs=800, alpha=0.001):
        for i in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)


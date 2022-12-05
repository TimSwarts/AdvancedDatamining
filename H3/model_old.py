import math
import random
from collections import Counter
from copy import deepcopy


# noinspection DuplicatedCode
class Perceptron:
    def __init__(self, dim):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for _ in range(dim)]
        self.predictions = []
        self.check = 0

    def __repr__(self):
        text = f'Perceptron(dim={self.dim})'
        return text

    def predict(self, xs):
        self.predictions = [1 if self.bias + sum(self.weights[i] * point[i] for i in range(self.dim)) > 0 else -1 if
                            self.bias + sum(self.weights[i] * point[i] for i in range(self.dim)) < 0 else 0 for point
                            in xs]
        return self.predictions

    def partial_fit(self, xs, ys):
        i = 0
        for x, y in zip(xs, ys):
            self.bias = self.bias - (self.predictions[i] - y)
            self.weights = [self.weights[j] - (self.predictions[i] - y) * x[j] for j in range(self.dim)]
            self.predict(xs)
            i += 1

    def fit(self, xs, ys, epochs=0):
        if not epochs == 0:
            for i in range(epochs):
                self.partial_fit(xs, ys)
                self.predict(xs)
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
        text = f'LinearRegression(dim={self.dim})'
        return text

    def predict(self, xs):
        self.predictions = [self.bias + sum(self.weights[i] * point[i] for i in range(self.dim)) for point
                            in xs]
        return self.predictions

    def partial_fit(self, xs, ys, *, alpha=0.01):
        i = 0
        for x, y in zip(xs, ys):
            self.bias = self.bias - alpha * (self.predictions[i] - y)
            self.weights = [self.weights[j] - alpha * (self.predictions[i] - y) * x[j] for j in range(self.dim)]
            self.predict(xs)
            i += 1

    def fit(self, xs, ys, epochs=500, alpha=0.01):
        for i in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)
            self.predict(xs)


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
        self.weights = [0 for _ in range(dim)]

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


class Layer:

    classcounter = Counter()

    def __init__(self, outputs, *, name=None, next=None):
        Layer.classcounter[type(self)] += 1
        if name is None:
            name = f'{type(self).__name__}_{Layer.classcounter[type(self)]}'
        self.inputs = 0
        self.outputs = outputs
        self.name = name
        self.next = next

    def __repr__(self):
        text = f'Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __add__(self, next):
        result = deepcopy(self)
        result.add(deepcopy(next))
        return result

    def __getitem__(self, index):
        if index == 0 or index == self.name:
            return self
        if isinstance(index, int):
            if self.next is None:
                raise IndexError('Layer index out of range')
            return self.next[index - 1]
        if isinstance(index, str):
            if self.next is None:
                raise KeyError(index)
            return self.next[index]
        raise TypeError(f'Layer indices must be integers or strings, not {type(index).__name__}')

    def add(self, next):
        if self.next is None:
            self.next = next
            next.set_inputs(self.outputs)
        else:
            self.next.add()

    def set_inputs(self, inputs):
        self.inputs = inputs

    def __call__(self, xs):
        raise NotImplementedError('Abstract __call__ method')


class InputLayer(Layer):

    def __repr__(self):
        text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        return self.next(xs, ys, alpha)

    def set_inputs(self, inputs):
        raise NotImplementedError()

    def predict(self, xs):
        yhats, _ = self(xs)
        return yhats

    def partial_fit(self, xs, ys, alpha=0.001):
        self(xs, ys, alpha)

    def fit(self, xs, ys, epochs=800, alpha=0.001):
        for i in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)

    def evaluate(self, xs, ys):
        _, ls = self(xs, ys)
        lmean = sum(ls) / len(ls)
        return lmean


class DenseLayer(Layer):
    def __init__(self, outputs, *, name=None, next=None):
        super().__init__(outputs, name=name, next=next)
        self.bias = [0.0 for _ in range(self.outputs)]
        self.weights = None

    def __repr__(self):
        text = f'DenseLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        aa = []   # Uitvoerwaarden voor alle instances xs
        for x in xs:
            a = []   # Uitvoerwaarde voor één instance x
            for o in range(self.outputs):
                # Bereken voor elk neuron o uit de lijst invoerwaarden x de uitvoerwaarde
                value = self.bias[o] + sum(self.weights[o][i] * x[i] for i in range(self.inputs))
                a.append(value)
            aa.append(a)
        yhats, ls, gs = self.next(aa, ys)  # 𝑔𝑛𝑖=∑𝑜𝑤𝑜𝑖⋅𝑞𝑛𝑜
        # Set gs
        gs_new = []
        for i in range(self.inputs):
            gs_new.append(sum(self.weights[o][i] * gs[i][o] for o in self.outputs))

        # change bias and weights
        for n, x in enumerate(xs):
            for o in self.outputs:
                self.bias[o] -= alpha/(sum(ls)/len(ls)) * gs[n][o]
                for i in range(self.inputs):
                    pass

        return yhats, ls, gs_new

    def set_inputs(self, inputs):
        self.inputs = inputs
        value = math.sqrt(6 / (self.inputs + self.outputs))
        if not self.weights:
            self.weights = [[random.uniform(-value, value) for _ in range(self.inputs)] for _ in range(self.outputs)]


class ActivationLayer(Layer):
    def __init__(self, outputs, *, name=None, next=None, activation=linear):
        super().__init__(outputs, name=name, next=next)
        self.activation = activation

    def __repr__(self):
        text = f'ActivationLayer(outputs={self.outputs}, name={repr(self.name)}, activation={self.activation.__name__})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        hh = []   # Uitvoerwaarden voor alle instances xs
        for x in xs:
            h = []   # Uitvoerwaarde voor één instance x
            for o in range(self.outputs):
                # Bereken voor elk neuron o uit de lijst invoerwaarden x de uitvoerwaarde
                value = self.activation(x[o])
                h.append(value)
            hh.append(h)
        yhats, ls, gs = self.next(hh, ys, alpha)  # 𝑔𝑛𝑖=𝜑′(𝑥𝑛𝑖)⋅𝑞𝑛𝑖
        # Set gs
        if alpha is not None:
            gs_new = []
            for n, x in enumerate(xs):
                for i in range(self.inputs):
                    gs_new.append(self.activation(x[i]) * gs[n][i])
        return yhats, ls, gs


class LossLayer(Layer):
    def __init__(self, loss=mean_squared_error, name=None):
        self.loss = loss
        self.delta_loss = derivative(loss)
        Layer.classcounter[type(self)] += 1
        if name is None:
            name = f'{type(self).__name__}_{Layer.classcounter[type(self)]}'
        self.name = name

    def __repr__(self):
        text = f'LossLayer(name={repr(self.name)}, loss={self.loss.__name__})'
        return text

    def __call__(self, xs, ys=None, alpha=None):
        yhats = xs
        ls = None
        gs = None
        if ys is not None:
            ls = []
            for yhat, y in zip(yhats, ys):
                ls.append(sum(self.loss(yhat[o], y[o]) for o in range(self.inputs)))
        if alpha is not None:
            gs = []
            for x, y in zip(xs, ys):
                g = [self.delta_loss(x[i], y[i]) for i in range(self.inputs)]
                gs.append(g)
        return yhats, ls, gs

    def add(self, next):
        raise NotImplementedError()

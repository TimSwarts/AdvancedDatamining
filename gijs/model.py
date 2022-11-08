import math
from collections import Counter
from copy import deepcopy
import random


def linear(a):
    return a


def sign(y):
    if y > 0:
        y = 1
    elif y < 0:
        y = -1
    return y


def softsign(a):
    return a / (1 + abs(a))


def mean_squared_error(yhat, y):
    return (yhat-y)**2


def mean_absolute_error(yhat, y):
    return abs(yhat-y)


def tanh(x):
    if abs(x) >=700:
        return abs(x)/x
    else:
        return (math.e**x - math.e**-x) / (math.e**x + math.e**-x)


def hinge(yhat, y):
    return max(1-yhat*y, 0)


def derivative(function, delta=0.001):

    def wrapper_derivative(x, *args):
        if 2*delta*x == 0:
            y = x+delta
            return (function(y + delta * y, *args) - function(y - delta * y, *args)) / (2 * delta * y)
        else:
            return (function(x+delta*x, *args) - function(x-delta*x, *args)) / (2*delta*x)

    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    return wrapper_derivative


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

    def __call__(self, xs):
        raise NotImplementedError('Abstract __call__ method')

    def add(self, next):
        # TODO: should prevent adding an already added neuron layer (prevents looping)
        if self.next is None:
            self.next = next
            next.set_inputs(self.outputs)
        else:
            self.next.add(next)

    def set_inputs(self, inputs):
        self.inputs = inputs


class InputLayer(Layer):

    def __repr__(self):
        text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        return self.next(xs, ys, alpha)

    def predict(self, xs):
        yhats, _, _ = self(xs)
        return yhats

    def evaluate(self, xs, ys):
        _, ls, _ = self(xs, ys)
        lmean = sum(ls) / len(ls)
        return lmean

    def set_inputs(self, inputs):
        return NotImplementedError()

    def partial_fit(self, xs, ys, alpha=0.1):
        self(xs, ys, alpha)

    def fit(self, xs, ys, *, alpha=10**-4, epochs=4000):
        while epochs > 0:
            epochs -= 1
            self.partial_fit(xs, ys, alpha=alpha)


class DenseLayer(Layer):

    def __init__(self, outputs, *, name=None, next=None):
        Layer.__init__(self, outputs, name=name, next=next)
        self.bias = [0 for i in range(outputs)]
        self.weights = [[] for i in range(outputs)]

    def __repr__(self):
        text = f'DenseLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        aa = []  # Uitvoerwaarden voor alle instances xs
        for x in xs:
            a = []  # Uitvoerwaarde voor één instance x
            for o in range(self.outputs):
                # Bereken voor elk neuron o uit de lijst invoerwaarden x de uitvoerwaarde
                ao = self.bias[o]
                for i in range(self.inputs):
                    ao += self.weights[o][i] * x[i]
                a.append(ao)
            aa.append(a)

        yhats, ls, gs = self.next(aa, ys, alpha)

        # gs manipulation
        if ys and alpha is not None:
            newgs = []
            for g, x in zip(gs, xs):
                gni = []
                for i in range(self.inputs):
                    gni_o = 0
                    for o in range(self.outputs):
                        gni_o += self.weights[o][i] * g[o]
                        self.bias[o] -= (alpha / len(xs)) * g[o]
                        self.weights[o][i] -= (alpha / len(xs)) * (g[o] * x[i])
                    gni.append(gni_o)
                newgs.append(gni)
            gs = newgs
        return yhats, ls, gs

    def set_inputs(self, inputs):
        self.inputs = inputs
        for x in range(self.outputs):
            for i in range(inputs):
                random_range = math.sqrt(6 / (inputs + self.outputs))
                weight = random.uniform(-random_range, random_range)
                self.weights[x-1].append(weight)


class ActivationLayer(Layer):

    def __init__(self, outputs, *, name=None, next=None, activation=linear):
        Layer.__init__(self, outputs, name=name, next=next)
        self.activation = activation

    def __repr__(self):
        text = f'ActivationLayer(outputs={self.outputs}, name={repr(self.name)}' \
            f', activation={repr(self.activation.__name__)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        hh = []  # Uitvoerwaarden voor alle instances xs
        for x in xs:
            h = []  # Uitvoerwaarde voor één instance x
            for o in range(self.outputs):
                # Bereken voor elk neuron o uit de lijst invoerwaarden x de uitvoerwaarde
                ho = self.activation(x[o])
                h.append(ho)
            hh.append(h)
        yhats, ls, gs = self.next(hh, ys, alpha)

        # gs
        if gs is not None:
            newgs = []
            delta_activation = derivative(self.activation)
            for x, g in zip(xs, gs):
                newgs.append([delta_activation(x[i])*g[i] for i in range(self.inputs)])
            gs = newgs

        return yhats, ls, gs


class SoftmaxLayer(Layer):
    # wat wordt er bedoelt met geen parameters? Want hij heeft nog wel input voor init nodig en krijgt hij ook

    def __init__(self, outputs, *, name=None, next=None, activation=softsign):
        Layer.__init__(self, outputs, name=name, next=next)
        self.activation = activation

    def __repr__(self):
        text = f'SoftmaxLayer(outputs={self.outputs}, name={repr(self.name)}' \
            f', activation={repr(self.activation.__name__)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        hh = []  # Uitvoerwaarden voor alle instances xs
        for x in xs:
            h = []  # Uitvoerwaarde voor één instance x

            # prevent oferflow by removing max value from every instance
            x = [i-max(x) for i in x]

            for o in range(self.outputs):
                # Bereken voor elk neuron o uit de lijst invoerwaarden x de uitvoerwaarde
                h.append(math.e**x[o])
            sum_h = sum(h)
            hh.append([x/sum_h for x in h])
        yhats, ls, gs = self.next(hh, ys, alpha)

        # gs
        if gs is not None:
            delta_act = derivative(self.activation)
            newgs = []
            for yhat, g, x in zip(yhats, gs, xs):
                for i in range(self.inputs):
                    gg = []
                    for o in range(self.outputs):
                        gg.append(g[o] * yhat[o] * ((i == o)-yhat[i]))
                    newgs.append(gg)
            gs = newgs
        return yhats, ls, gs


class LossLayer(Layer):

    def __init__(self, loss=mean_squared_error, name=None):
        Layer.__init__(self, 0, name=name, next=None)
        self.loss = loss
        self.delta_loss = derivative(loss)

    def __repr__(self):
        text = f'LossLayer(outputs={self.outputs}, name={repr(self.name)}' \
            f', loss={repr(self.loss.__name__)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        yhats = xs
        ls = None
        gs = None
        if ys is not None:
            ls = []
            for x, y in zip(xs, ys):
                l = sum(self.loss(x[i], y[i]) for i in range(self.inputs))
                ls.append(l)

            if alpha is not None:
                gs = []
                for x, y in zip(xs, ys):
                    g = [self.delta_loss(x[i], y[i]) for i in range(self.inputs)]
                    gs.append(g)
        return yhats, ls, gs

    def add(self, next):
        return NotImplementedError()


class Neuron:

    def __init__(self, dim, activation=linear, loss=mean_squared_error):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for d in range(dim)]
        self.activation = activation
        self.loss = loss

    def __repr__(self):
        text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
        return text

    def predict(self, xs):
        """

        :param xs: lijst van lijsten met instances
        :return: een lijst aan predicties
        """
        # list = [[x1,x2], [x1, x2], ...]
        # w and b should be 0 without training
        output = []
        for point in xs:
            y = 0
            for i in range(self.dim):
                y += self.weights[i] * point[i]
            y += self.bias
            y = self.activation(y)
            output.append(y)
        return output

    def partial_fit(self, xs, ys, *, alpha=10**-5):
        y_predicts = self.predict(xs)
        derivative_loss = derivative(self.loss)
        derivative_activation = derivative(self.activation)

        for x, y, y_predict in zip(xs, ys, y_predicts):
            self.bias -= alpha * derivative_loss(y_predict, y) * derivative_activation(y_predict)
            for i in range(self.dim):
                self.weights[i] -= alpha * derivative_loss(y_predict, y) * derivative_activation(y_predict)*x[i]

    def fit(self, xs, ys, *, alpha=10**-4, epochs=4000):
        """

        :param xs: list with dimensions of a point
        :param ys: list with group of that point
        :param epochs: The ammount of epochs. If 0 it will continue until there is no more change.
        :param alpha: the learning rate
        :return: nothing
        """
        while epochs > 0:
            epochs -= 1
            self.partial_fit(xs, ys, alpha=alpha)


class Perceptron:
    
    def __init__(self, dim):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for d in range(dim)]

    def __repr__(self):
        text = f'Perceptron(dim={self.dim})'
        return text

    def sgn(self, y):
        if y > 0:
            y = 1
        elif y < 0:
            y = -1
        return y

    def predict(self, xs):
        """

        :param xs: lijst van lijsten met instances
        :return: een lijst aan predicties
        """
        # list = [[x1,x2], [x1, x2], ...]
        # w and b should be 0 without training
        output = []
        for point in xs:
            y = 0
            for i in range(self.dim):
                y += self.weights[i] * point[i]
            y += self.bias
            y = self.sgn(y)
            output.append(y)
        return output

    def partial_fit(self, xs, ys):
        """

        :param xs: list with dimensions of a point
        :param ys: list with group of that point
        :return: nothing
        """
        # for every
        y_predicts = self.predict(xs)
        for x, y, y_predict in zip(xs, ys, y_predicts):
            self.bias -= (y_predict - y)
            for i in range(self.dim):
                self.weights[i] -= (y_predict - y)*x[i]

    def fit(self, xs, ys, *, epochs=0):
        """

        :param xs: list with dimensions of a point
        :param ys: list with group of that point
        :param epochs: The ammount of epochs. If 0 it will continue until there is no more change.
        :return: nothing
        """
        if epochs != 0:
            while epochs > 0:
                epochs -= 1
                self.partial_fit(xs, ys)
        else:
            repeat = True
            while repeat:
                prev_bias = self.bias
                prev_weights = self.weights
                self.partial_fit(xs, ys)
                if self.bias == prev_bias and self.weights == prev_weights:
                    repeat = False


class LinearRegression:

    def __init__(self, dim):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for d in range(dim)]

    def __repr__(self):
        text = f'LinearRegression(dim={self.dim})'
        return text

    def predict(self, xs):
        """

        :param xs: lijst van lijsten met instances
        :return: een lijst aan predicties
        """
        # list = [[x1,x2], [x1, x2], ...]
        # w and b should be 0 without training
        output = []
        for point in xs:
            y = 0
            for i in range(self.dim):
                y += self.weights[i] * point[i]
            y += self.bias
            output.append(y)
        return output

    def partial_fit(self, xs, ys, *, alpha=0.001):
        """

        :param xs: list with dimensions of a point
        :param ys: list with group of that point
        :return: nothing
        """
        # for every
        y_predicts = self.predict(xs)
        for x, y, y_predict in zip(xs, ys, y_predicts):
            self.bias -= alpha*(y_predict - y)
            for i in range(self.dim):
                self.weights[i] -= alpha*(y_predict - y) * x[i]

    def fit(self, xs, ys, *, alpha=0.5, epochs=500):
        """

        :param xs: list with dimensions of a point
        :param ys: list with group of that point
        :param epochs: The ammount of epochs. If 0 it will continue until there is no more change.
        :param alpha: the learning rate
        :return: nothing
        """
        while epochs > 0:
            epochs -= 1
            self.partial_fit(xs, ys)

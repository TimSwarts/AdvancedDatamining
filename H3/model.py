import math
import random
from collections import Counter
from copy import deepcopy


class Perceptron:
    def __init__(self, dim):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for _ in range(dim)]
        self.predictions = []

    def __repr__(self):
        text = f'Perceptron(dim={self.dim})'
        return text

    def predict(self, xs):
        self.predictions = [1 if self.bias + sum(self.weights[i] * point[i] for i in range(self.dim)) > 0 else -1 if
                            self.bias + sum(self.weights[i] * point[i] for i in range(self.dim)) < 0 else 0
                            for point in xs]
        return self.predictions

    def predict2(self, xs):
        self.predictions = []

        for point in xs:
            prediction = self.bias + sum(self.weights[i] * point[i] for i in range(self.dim))
            if prediction > 0:
                self.predictions.append(1)
            elif prediction < 0:
                self.predictions.append(-1)
            else:
                self.predictions.append(0)

        return self.predictions

    def partial_fit(self, xs, ys):
        i = 0
        self.predict(xs)
        for x, y in zip(xs, ys):
            self.bias = self.bias - (self.predictions[i] - y)
            self.weights = [self.weights[j] - (self.predictions[i] - y) * x[j] for j in range(self.dim)]
            self.predict(xs)
            i += 1

    def fit(self, xs, ys, epochs=0):
        if epochs != 0:
            for _ in range(epochs):
                self.partial_fit(xs, ys)
        else:
            repeat = True
            epn = 0
            while repeat:
                prev_bias = self.bias
                prev_weights = self.weights
                self.partial_fit(xs, ys)
                epn += 1
                if self.bias == prev_bias and self.weights == prev_weights:
                    repeat = False
                    print(f'number of epochs needed for convergence: {epn}')


class LinearRegression:
    def __init__(self, dim):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for _ in range(dim)]
        self.predictions = []

    def __repr__(self):
        text = f'LinearRegression(dim={self.dim})'
        return text

    def predict(self, xs):
        self.predictions = [self.bias + sum(self.weights[i] * point[i] for i in range(self.dim)) for point in xs]
        return self.predictions

    def partial_fit(self, xs, ys, alpha=0.01):
        i = 0
        self.predict(xs)
        for x, y in zip(xs, ys):
            self.bias = self.bias - alpha * (self.predictions[i] - y)
            self.weights = [self.weights[j] - alpha * (self.predictions[i] - y) * x[j] for j in range(self.dim)]
            self.predict(xs)
            i += 1

    def fit(self, xs, ys, alpha=0.01, epochs=500):
        for _ in range(epochs):
            self.partial_fit(xs, ys, alpha)


# Activation functions
def linear(a):
    """
    Identity function
    :param a: preactivation value
    :return a: postactivation value
    """
    return a


def sign(a):
    """
    Signum function
    :param a: preactivation value
    :return: post activation value
    """
    if a > 0:
        return 1
    if a < 0:
        return -1
    else:
        return 0


def tanh(a):
    """
    Tangent hyperbolic function
    :param a: preactivation value
    :return: post activation value
    """
    return (math.e ** a - math.e ** - a) / (math.e ** a + math.e ** - a)


# Loss functions
def mean_squared_error(yhat, y):
    """
    Mean squared loss function, calculates loss
    """
    return (yhat - y) ** 2


def mean_absolute_error(yhat, y):
    """
    Mean absolute loss function, calculates loss
    """
    return abs(yhat - y)


def hinge(yhat, y):
    """
    Hinge loss function
    """
    return max(1 - yhat * y, 0)  # max(1−𝑦̂ ⋅𝑦,0)


# Derivative function calculator
def derivative(function, delta=0.01):
    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)

    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    return wrapper_derivative


# Neuron class
class Neuron:
    def __init__(self, dim=2, activation=linear, loss=mean_squared_error):
        self.dim = dim
        self.activation = activation
        self.loss = loss
        self.bias = 0
        self.weights = [0 for _ in range(dim)]
        print("successful initialisation of neuron")

    def __repr__(self):
        text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
        return text

    def predict(self, xs):
        """
        This function calculates the yhat value for every point in a list of attribute values
        using the neuron prediction model: 𝑦̂ = 𝜑(𝑏 + ∑𝑖 𝑤𝑖⋅𝑥𝑖)
        :param xs: This should be a list of lists wherein each sub-list should have length self.dim
        :return predictions: The final list of yhat values with length len(xs)
        """
        # empty predictions
        predictions = []

        for point in xs:
            # calculate the pre activation value: b + sum(wi * xi)
            pre_activation = self.bias + sum(self.weights[i] * point[i] for i in range(self.dim))
            # calculate the post activation value: phi(a)
            post_activation = self.activation(pre_activation)
            # add value to prediction list
            predictions.append(post_activation)

        return predictions

    def partial_fit(self, xs, ys, *, alpha=0.01):
        # get predictions
        predictions = self.predict(xs)
        # one epoch consists of an update for every instance
        for x, y, yhat in zip(xs, ys, predictions):
            # update bias with: b <- b - alpha * derivative(loss) * derivative(activation)
            self.bias = self.bias - alpha * derivative(self.loss)(yhat, y) * derivative(self.activation)(yhat)

            # update weights with: wi <- wi - alpha * derivative(loss) * derivative(activation)
            self.weights = [self.weights[i] - alpha * derivative(self.loss)(yhat, y) * derivative(self.activation)(yhat)
                            * x[i] for i in range(self.dim)]

    def fit(self, xs, ys, epochs=800, alpha=0.001):
        for _ in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)


class Layer:
    classcounter = Counter()

    def __init__(self, outputs, *, name=None, next=None):
        Layer.classcounter[type(self)] += 1
        if name is None:
            name = f'{type(self).__name__}_{Layer.classcounter[type(self)]}'  # example: Layer_1
        self.inputs = 0
        self.outputs = outputs
        self.name = name
        self.next = next

    def __repr__(self):
        text = f'Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def add(self, next):
        """
        This function adds layers to the network, a new layer is saved in the self.next variable of the current layer
        or in that of the next layer if self.next is not empty.
        :param next: A new layer that will be added at the end of the network.
        """
        if self.next is None:
            self.next = next
            next.set_inputs(self.outputs)
        else:
            self.next.add(next)

    def set_inputs(self, inputs):
        self.inputs = inputs

    # optional __add__ function to allow usage of + operator to add layers
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


class InputLayer(Layer):

    def __repr__(self):
        text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def set_inputs(self, inputs):
        raise NotImplementedError("An InputLayer itself can not receive inputs from previous layers,"
                                  "as it is always the first layer of a network.")

    def __call__(self, xs):
        return self.next(xs)

    def predict(self, xs):
        yhats = self(xs)
        return yhats


class DenseLayer(Layer):
    def __init__(self, outputs, *, name=None, next=None):
        super().__init__(outputs, name=name, next=next)
        # Set biases, one biases for every neuron (equal to the amount of outputs)
        self.bias = [0 for _ in range(self.outputs)]

        # Initialise weights (filled later in set_inputs method)
        self.weights = None

    def __repr__(self):
        text = f'DenseLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def set_inputs(self, inputs):
        self.inputs = inputs
        limit = math.sqrt(6 / (self.inputs + self.outputs))
        if not self.weights:
            self.weights = [[random.uniform(-limit, limit) for _ in range(self.inputs)] for _ in range(self.outputs)]

    def __call__(self, xs):
        """
        xs should be a list of lists of values, where each sublist has a number of values equal to self.inputs
        """
        aa = []   # Uitvoerwaarden voor alle instances xs (xs is een (nested) lijst met instances)
        for x in xs:
            a = []   # Uitvoerwaarde voor één instance x (x is een lijst met attributen)
            for o in range(self.outputs):
                # Bereken voor elk neuron o uit de lijst invoerwaarden x de uitvoerwaarde
                pre_activation = self.bias[o] + sum(self.weights[o][i] * x[i] for i in range(self.inputs))
                a.append(pre_activation)  # a is lijst met de output waarden van 1 instance
            aa.append(a)  # aa is een nested lijst met de output waarden van alle instances
        yhats = self.next(aa)

        # concept:
        values = [[self.bias[o] + sum(self.weights[o][i] * x[i] for i in range(self.inputs))
               for o in self.outputs]
              for x in xs]
        return yhats


class ActivationLayer(Layer):
    def __init__(self, outputs, *, name=None, next=None, activation=linear):
        super().__init__(outputs, name=name, next=next)
        self.activation = activation

    def __repr__(self):
        text = f'ActivationLayer(outputs={self.outputs}, name={self.name}, activation={self.activation.__name__})'
        return text

    def __call__(self, aa):
        hh = []   # Uitvoerwaarden voor alle pre activatie waarden berekend in de vorige laag
        for a in aa:
            h = []   # Uitvoerwaarde voor één pre activatie waarde
            for o in range(self.outputs):
                # Bereken voor elk neuron o uit de lijst invoerwaarden x de uitvoerwaarde
                post_activation = self.activation(a)
                h.append(post_activation)
            hh.append(h)
        yhats = self.next(hh)
        return yhats
class LossLayer(Layer):
    def __init__(self, loss=mean_squared_error, name=None):
        super().__init__(outputs=None, name=name)
        self.loss = loss

    def __repr__(self):
        text = f'LossLayer(loss={self.loss.__name__}, name={self.name})'
        return text

    def add(self, next):
        raise NotImplementedError("It is not possible to add a layer to a LossLayer,"
                                  "since a network should always end with a single LossLayer")



def main():
    """
    main function used for testing
    """
    # create perceptron
    perceptron = Perceptron(2)
    # check correct initialisation
    print(perceptron)
    # get some test data
    xs = [[-1, -1], [-1, 1], [1, -1], [1, 1]]
    ys = [-1, -1, -1, 1]
    # train the perceptron
    perceptron.fit(xs, ys)
    # print results
    print(f'bias={perceptron.bias}\n'
          f'weights={perceptron.weights}\n\n'
          f'yhat: {perceptron.predictions}\n'
          f'y:    {ys}')

    my_network = Layer(outputs=3, name='Input')
    my_network.add(Layer(outputs=2, name='Hidden'))
    my_network.add(Layer(outputs=1, name='Output'))
    print(my_network['Output'])
    return 0


if __name__ == "__main__":
    main()

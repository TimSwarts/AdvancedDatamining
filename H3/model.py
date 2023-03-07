import math
import random
from collections import Counter
from copy import deepcopy

import data


# Perceptron and LinearRegression classes
class Perceptron:
    def __init__(self, dim):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0.0 for _ in range(dim)]

    def __repr__(self):
        text = f'Perceptron(dim={self.dim})'
        return text

    def predict(self, xs, /):
        predictions = [1 if self.bias + sum(self.weights[i] * point[i] for i in range(self.dim)) > 0 else -1 if
                            self.bias + sum(self.weights[i] * point[i] for i in range(self.dim)) < 0 else 0
                            for point in xs]
        return predictions

    def predict2(self, xs):
        predictions = []

        for point in xs:
            prediction = self.bias + sum(self.weights[i] * point[i] for i in range(self.dim))
            if prediction > 0:
                predictions.append(1)
            elif prediction < 0:
                predictions.append(-1)
            else:
                predictions.append(0)

        return predictions

    def partial_fit(self, xs, ys):
        for x, y in zip(xs, ys):
            # Get prediction of current x
            yhat = self.predict([x])[0]
            # Update bias
            self.bias = self.bias - (yhat - y)
            # Update weights
            self.weights = [self.weights[j] - (yhat - y) * x[j] for j in range(self.dim)]

    def fit(self, xs, ys, *, epochs=0):
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

    def __repr__(self):
        text = f'LinearRegression(dim={self.dim})'
        return text

    def predict(self, xs):
        predictions = [self.bias + sum(self.weights[i] * point[i] for i in range(self.dim)) for point in xs]
        return predictions

    def partial_fit(self, xs, ys, *, alpha=0.01):
        for x, y in zip(xs, ys):
            prediction = self.predict([x])[0]
            self.bias = self.bias - alpha * (prediction - y)
            self.weights = [self.weights[j] - alpha * (prediction - y) * x[j] for j in range(self.dim)]

    def fit(self, xs, ys, *, alpha=0.01, epochs=500):
        for _ in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)


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
def derivative(function, delta=0.001):
    """
    This function returns a function that calculates a numerical approximation of the slope in a point on the
    input function
    :param function: this is the function for which a derivative function is set up
    :param delta: this is the delta used as the difference between two points to approximate the derivative
    :return function: the derivative function of the input function
    """
    # Create a function that calculates a numerical approximation of the slope in a point on the given input function
    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)
    # Give it a distinct name
    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    # Return the wrapper function
    return wrapper_derivative


# Derivative function calculator
def derivative(function, delta=0.001):
    """
    This function returns a function that calculates a numerical approximation of the slope in a point on the
    input function
    :param function: This is the function for which a derivative function is set up (function)
    :param delta: This is the delta used as the difference between two points to approximate the derivative (float)
    :return function: The derivative function of the input function (function)
    """
    # Create a function that calculates a numerical approximation of the slope in a point on the given input function
    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2 * delta)
    # Give it a distinct name
    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    # Return the wrapper function
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
        # One epoch consists of an update for every instance
        for x, y in zip(xs, ys):
            # Calculate the pre activation value: b + sum(wi * xi)
            pre_activation = self.bias + sum(self.weights[i] * x[i] for i in range(self.dim))
            # Calculate the post activation(yhat) for this instance value: phi(a)
            yhat = self.activation(pre_activation)

            # Update bias with: b <- b - alpha * derivative(loss) * derivative(activation)
            self.bias = self.bias - alpha * derivative(self.loss)(yhat, y) * derivative(self.activation)(pre_activation)
            # Update weights with: wi <- wi - alpha * derivative(loss) * derivative(activation) * xi
            self.weights = [self.weights[i] -
                            alpha * derivative(self.loss)(yhat, y) * derivative(self.activation)(pre_activation) * x[i]
                            for i in range(self.dim)]

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
        """
        This function makes the network iterable
        """
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

    def __call__(self, xs, ys=None, alpha=None):
        return self.next(xs, ys, alpha)

    def predict(self, xs):
        yhats, _, _ = self(xs)
        return yhats

    def evaluate(self, xs, ys):
        _, ls, _ = self(xs, ys)
        lmean = sum(ls) / len(ls)
        return lmean

    def partial_fit(self, xs, ys, alpha=0.001):
        self(xs, ys, alpha)

    def fit(self, xs, ys, epochs=800, alpha=0.001):
        for _ in range(epochs):
            self.partial_fit(xs, ys, alpha=alpha)

class DenseLayer(Layer):
    def __init__(self, outputs, *, name=None, next=None):
        super().__init__(outputs, name=name, next=next)
        # Set biases, one bias for every neuron (equal to the amount of outputs)
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

    def __call__(self, xs, ys=None, alpha=None):
        """
        xs should be a list of lists of values, where each sublist has a number of values equal to self.inputs
        """
        aa = []   # Uitvoerwaarden voor alle instances xs (xs is een (nested) lijst met instances)
        gxs = None # Set gradient from loss to x input equal to None
        for x in xs:
            a = []   # Uitvoerwaarde voor één instance x (x is een lijst met attributen)
            for o in range(self.outputs):
                # Bereken voor elk neuron o uit de lijst invoerwaarden x de uitvoerwaarde
                pre_activation = self.bias[o] + sum(self.weights[o][i] * x[i] for i in range(self.inputs))
                a.append(pre_activation)  # a is lijst met de output waarden van 1 instance
            aa.append(a)  # aa is een nested lijst met de output waarden van alle instances

        # Send to aa next layer, and collect it's yhats, ls, and gas
        yhats, ls, gas = self.next(aa, ys, alpha)

        if alpha:
            # Initiate empty list
            gxs = []
            # Calculate gradient vectors for all instances
            for x, gan in zip(xs, gas):
                gxn = [sum(self.weights[o][i] * gan[o] for o in range(self.outputs)) for i in range(self.inputs)]
                # Add instance to list
                gxs.append(gxn)
                # Update bias and weights per instance
                for o in range(self.outputs):
                    # b <- b - alpha/N * xi * d_ln/d_ano
                    self.bias[o] = self.bias[o] - alpha/len(xs) * gan[o]
                    # w <- w - alpha/N * xi * d_ln/d_ano
                    self.weights[o] = [self.weights[o][i] - alpha/len(xs) * gan[o] * x[i] for i in range(self.inputs)]

        return yhats, ls, gxs


class ActivationLayer(Layer):
    def __init__(self, outputs, *, name=None, next=None, activation=linear):
        super().__init__(outputs, name=name, next=next)
        self.activation = activation

    def __repr__(self):
        text = f'ActivationLayer(outputs={self.outputs}, name={self.name}, activation={self.activation.__name__})'
        return text

    def __call__(self, aa, ys=None, alpha=None):
        hh = []   # Uitvoerwaarden voor alle pre activatie waarden berekend in de vorige laag
        gas = None # Set gradient from loss to pre activation value to None
        for a in aa:
            h = []   # Uitvoerwaarde voor één pre activatie waarde
            for o in range(self.outputs):
                # Bereken voor elk neuron o uit de lijst invoerwaarden x de uitvoerwaarde
                post_activation = self.activation(a[o])
                h.append(post_activation)
            hh.append(h)

        # sent hh to the next layer and collext its yhats, ls, and gls
        yhats, ls, gls = self.next(hh, ys, alpha)

        if alpha:
            # calculate gradients from the loss to the pre_activation value
            gas = []

            # Calculate gradient vectors for all instances:, using the derivative of the activation function and gls
            for a, gln in zip(aa, gls):
                # For each instance, calculate the gradient from the loss to each pre activation value
                gan = [derivative(self.activation)(a[i]) * gln[i] for i in range(self.inputs)]
                gas.append(gan)


        return yhats, ls, gas


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

    def __call__(self, hh, ys=None, alpha=None):
        # yhats is the output of the previous layer, because the loss layer is always last
        yhats = hh
        # ls, the loss, which will be a list of losses for all outputs in yhats, starts at None
        ls = None
        # gls, will be list of gradient vectors, one for each instance, with one value for each output of the prev layer
        # starts None
        gls = None
        if ys:
            ls = []
            # For all instances calculate loss:
            for yhat, y in zip(yhats, ys):
                # Take sum of the loss of all outputs(number of outputs previous layer=inputs this layer)
                ln = sum(self.loss(yhat[o], y[o]) for o in range(self.inputs))
                ls.append(ln)

            # If there is a learning rate
            if alpha:
                gls = []
                # Calculate a gradient vectors for all instances in yhats
                for yhat, y in zip(yhats, ys):
                    # Each instance can have multiple outputs, with the derivative of the loss we calculate dl/dyhat
                    gln = [derivative(self.loss)(yhat[o], y[o]) for o in range(self.inputs)]
                    gls.append(gln)
        return yhats, ls, gls




def main():
    """
    main function used for testing
    """
    my_network = InputLayer(2) + \
                 DenseLayer(2) + \
                 ActivationLayer(2, activation=sign) + \
                 DenseLayer(1) + \
                 LossLayer()
    my_network[1].bias = [1.0, -1.0]
    my_network[1].weights = [[1.0, 1.0], [1.0, 1.0]]
    my_network[3].bias = [-1.0]
    my_network[3].weights = [[1.0, -1.0]]

    xs, ys = data.xorproblem()
    print(f"xs:  {xs}")
    yhats = my_network.predict(xs)
    return 0


if __name__ == "__main__":
    main()

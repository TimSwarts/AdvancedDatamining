from math import e, sqrt, log, exp, tanh as m_tanh
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
    :param a: preactivation value (float)
    :return a: postactivation value (float)
    """
    return a


def sign(a):
    """
    Signum function, this function gives -1 for all negative inputs, 1 for all positive inputs, and 0 for 0.
    :param a: preactivation value (float)
    :return: post activation value (float)
    """
    if a > 0:
        return 1
    if a < 0:
        return -1
    return 0


def tanh(a):
    """
    Tangent hyperbolic function, this gives output valies between -1 and 1.
    :param a: preactivation value (float)
    :return: post activation value (float)
    """
    return m_tanh(a)


def softsign(a):
    """
    Softsign function, this gives output valies between -1 and 1.
    :param a: preactivation value (float)
    :return: post activation value (float)
    """
    return a / (1 + abs(a))


def sigmoid(a):
    """
    Sigmoid function, this gives output values between 0 and 1.
    :param a: preactivation value (float)
    :return: post activation value (float)
    """
    try:
        return exp(a) / (1 + exp(a))
    except OverflowError:
        return 1.0


def softplus(a):
    """
    Softplus function, this is a smooth approximation to the ReLU function
    :param a: preactivation value (float)
    :return: post activation value (float)
    """
    try:
        return log(1 + e**a)
    except OverflowError:
        return a


def relu(a):
    """
    ReLU function, 0 for all negative inputs, identity function for all postive inputs
    :param a: preactivation value (float)
    :return: post activation value (float)
    """
    return max(0, a)


def swish(a, *, beta=1):
    """
    Swish function, or sigmoid linear unit
    The Swish function is defined as f(x) = x * sigmoid(beta*x)
    where sigmoid is the logistic sigmoid function and beta is a parameter that can be set(usually 1)
    or trained(in rare cases).
    :param a: preactivation value (float)
    :param beta: parameter to control the leaning of the function towards sigmoid or identity function. (float)
    :return: post activation value (float)
    """
    return a * sigmoid(a * beta)


# One hot encoded activation functions
def softmax(h):
    """
    This is the softmax function, it takes in a vector of numerical values h and converts it into
    a probability distribution
    :param h: a vector of numerical values (list[float] | tuple[float])
    :return y: a one-hot encoded probability distribution where each value represents the probability of a class
               associated with the index of that value. (list)
    """
    # Normalise h to prevent ZeroDivisionError and OverflowError, by subtracting the max from the vector
    h = [value - max(h) for value in h]
    # Calculate the sum of e^hi for every value hi in the list h, used as the denominator in the softmax function
    denominator = sum(e**hi for hi in h)
    # Apply softmax function to the list h
    y_hats = [e**ho / denominator for ho in h]
    # Return probability distribution
    return y_hats


# Loss functions
def mean_squared_error(yhat, y):
    """
    Mean squared loss function, calculates loss
    :param yhat: predicted value (float)
    :param y: real value (float)
    """
    return (yhat - y) ** 2


def mean_absolute_error(yhat, y):
    """
    Mean absolute loss function, calculates loss
    :param yhat: predicted value (float)
    :param y: real value (float)
    """
    return abs(yhat - y)


def hinge(yhat, y):
    """
    Hinge loss function, calculates loss
    :param yhat: predicted value (float)
    :param y: real value (float)
    :return: loss between 0 and 1 (float)
    """
    return max(1 - yhat * y, 0)  # max(1−𝑦̂ ⋅𝑦,0)


def binary_crossentropy(yhat, y, epsilon=0.0001):
    """
    Binary Cross Entropy loss function, calculates loss
    :param yhat: predicted value (float)
    :param y: real value (float)
    :param epsilon: minimum limit used for pseudo_log (float)
    :return: loss between 0 and 1 (float)
    """
    return -y * pseudo_log(yhat, epsilon) - (1 - y) * pseudo_log(1 - yhat, epsilon)


def categorical_crossentropy(yhat, y, epsilon=0.0001):
    """
    Categorical Cross Entropy loss function, calculates loss
    :param yhat: predicted value (float)
    :param y: real value (float)
    :param epsilon: minimum limit used for pseudo_log (float)
    :return: loss between 0 and 1 (float)
    """
    return -y * pseudo_log(yhat, epsilon)


# Support function
def pseudo_log(x, epsilon=0.001):
    """
    This function substitutes the log function in the cross entropy log functions, to prevent math domain errors.
    When the input of the log is smaller than a set minimum epsilon, a pseudo log is used instead. This prevents the
    cases where, due to underflow or during numerical differentiation, log(x) is attempted with x <= 0
    :param x: The input of the log (float | int)
    :param epsilon: The minimum limit (float)
    :return: log of x (float)
    """
    if x < epsilon:
        return log(epsilon) +  (x - epsilon)/epsilon
    return log(x)

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


# Neural network layers:
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

    def set_inputs(self, inputs: int):
        self.inputs = inputs
        limit = sqrt(6 / (self.inputs + self.outputs))
        if not self.weights:
            self.weights = [[random.uniform(-limit, limit) for _ in range(self.inputs)] for _ in range(self.outputs)]

    def __call__(self, xs, ys=None, alpha=None):
        """
        xs should be a list of lists of values, where each sublist has a number of values equal to self.inputs
        """
        aa = []   # Uitvoerwaarden voor alle instances xs (xs is een (nested) lijst met instances)
        gxs = None # Set gradient from loss to x-input equal to None
        for x in xs:
            a = []   # Uitvoerwaarde voor één instance x (x is een lijst met attributen)
            for o in range(self.outputs):
                # Bereken voor elk neuron o uit de lijst invoerwaarden x de uitvoerwaarde
                pre_activation = self.bias[o] + sum(self.weights[o][i] * x[i] for i in range(self.inputs))
                a.append(pre_activation)  # a is lijst met de output waarden van alle neuronen voor 1 instance
            aa.append(a)  # aa is een nested lijst met de output waarden van alle instances

        # Send to aa next layer, and collect its yhats, ls, and gas
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
                    # b <- b - alpha/N * xi * d_ln/d_ano (xi for bias = 1, therefor not included)
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
        hh = []   # Post activation values calculated from all pre activation values from the previous layer
        gas = None # Set gradient from loss to pre activation value to None
        for a in aa:
            h = []   # Post activation values of one instance
            for o in range(self.outputs):
                # Calculate post activation value for each neuron o, using the pre activation value (a)
                post_activation = self.activation(a[o])
                # Append post activation values of all neurons to data instance
                h.append(post_activation)
            # Collect all instances in the data set
            hh.append(h)

        # Send hh to the next layer and collect its yhats, ls, and gls
        yhats, ls, gls = self.next(hh, ys, alpha)

        if alpha:
            # Calculate gradients from the loss to the pre_activation value
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


class SoftmaxLayer(Layer):
    def __init__(self, outputs, *, name=None, next=None):
        super().__init__(outputs, name=name, next=next)

    def __repr__(self):
        text = f'SoftmaxLayer(outputs={self.outputs}, name={self.name})'
        return text

    def __call__(self, hh, ys=None, alpha=None):
        probs = []   # Final predictions
        ghs = None   # Set gradient from loss to h value to None
        for h in hh:
            prob = softmax(h)   # Probability distribution for one instance
            probs.append(prob)  # Collect all instances

        # Send probabilities to the next layer and collect its yhats, ls, and gls
        yhats, ls, gls = self.next(probs, ys, alpha)

        if alpha:
            # Calculate gradients from the loss to last layer neuron output values of all instances
            ghs = []
            for yhat, gln in zip(yhats, gls):
                # Calculate the gradient vectors for every instance, a vector contains the gradient from the loss to h
                ghn = [sum(gln[o] * yhat[o] * ((i==o) - yhat[i]) for o in range(self.outputs))
                       for i in range(self.inputs)]   # This uses the derivative of the softmax function: yno*(𝛿io-yni)
                ghs.append(ghn)


        return yhats, ls, ghs


def main():
    """
    main function used for testing
    """
    # my_network = InputLayer(2) + \
    #              DenseLayer(2) + \
    #              ActivationLayer(2, activation=sign) + \
    #              DenseLayer(1) + \
    #              LossLayer()
    # my_network[1].bias = [1.0, -1.0]
    # my_network[1].weights = [[1.0, 1.0], [1.0, 1.0]]
    # my_network[3].bias = [-1.0]
    # my_network[3].weights = [[1.0, -1.0]]
    #
    # xs, ys = data.xorproblem()
    # print(f"xs:  {xs}")
    # yhats = my_network.predict(xs)
    y = 1.0
    data.graph([categorical_crossentropy, binary_crossentropy], y, xlim=(0.0, 1.0))
    return 0


if __name__ == "__main__":
    main()

import math


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
                    print(epn)


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

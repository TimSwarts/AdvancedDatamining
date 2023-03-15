import math


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
        """
        This function trains the model for one epoch, updating the weights
        for every instance in the data set
        :param xs: This should be a list of lists wherein each sub-list should have length self.dim.
                   These are the input values of the model.
        :param ys: This is a list of labels (value: -1 or 1),
                   each label corresponding to a list of input values in xs
        """

        # Loop through all instances
        for x, y in zip(xs, ys):
            # Get prediction of current x
            yhat = self.predict([x])[0]
            # Update bias
            self.bias = self.bias - (yhat - y)
            # Update weights
            self.weights = [self.weights[j] - (yhat - y) * x[j] for j in range(self.dim)]

    def fit(self, xs, ys, *, epochs=0):
        """
        This function seeks to fully fit the model to the data set,
        it has an option to train for a given number of epochs, and an option to train until
        every instance is predicted correctly.
        :param xs: This should be a list of lists wherein each sub-list should have length self.dim.
                   These are the input values of the model.
        :param ys: This is a list of labels (value: -1 or 1),
                   each label corresponding to a list of input values in xs
        :param epochs: Keep this value at its default of 0 to keep training until full convergence
                       Set it to a preferred number of epochs to train for that amount.
        """
        # Set a boolean that checks if the while loop should continue repeating
        repeat = True
        # Initialise an epoch number counter with starting value 0
        epn = 0
        while repeat:
        # If repeat is true, keep fitting
            # Remember the current bias and weights
            prev_bias = self.bias
            prev_weights = self.weights

            # Call partial fit to update the bias and weights
            self.partial_fit(xs, ys)

            # This was one epoch, increase the counter by 1
            epn += 1

            if (self.bias == prev_bias and self.weights == prev_weights) or 0 < epochs == epn:
            # If the updated bias and weights are equal to the previous values
                # Then the model is no longer updating, because it is already predicting everything correctly
                repeat = False  # Set repeat to False in order to exit the while loop
                # Print how long it took to fit the model to the dataset
                print(f'number of epochs trained: {epn}')


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
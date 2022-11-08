class Perceptron():
    
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


def linear(a):
    return a


def sign(y):
    if y > 0:
        y = 1
    elif y < 0:
        y = -1
    return y

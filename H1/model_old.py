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
                self.fit(xs, ys)
            else:
                return


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

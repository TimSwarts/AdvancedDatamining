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


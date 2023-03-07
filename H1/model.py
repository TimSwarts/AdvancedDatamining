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


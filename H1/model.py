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
        # While repeat is true, keep fitting
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


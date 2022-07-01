import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Layer(object):
    def __init__(self, n_input, n_output, weights, bias, activation):

        self.weights = weights
        self.bias = bias
        self.activation = activation

    def activate(self, X):
        # X@W + b
        r = np.dot(X, self.weights) + self.bias

        self.activation_output = self._apply_activation(r)
        return self.activation_output

    # get the output of activation function
    def _apply_activation(self, r):
        if self.activation == 'relu':
            return np.maximum(r, 0)

        elif self.activation == 'tanh':
            return np.tanh(r)

        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-r))

        elif self.activation == 'step':
            return np.where(r >= 0, 1, 0)


class NeuralNetwork(object):
    def __init__(self):
        # layers list
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def feed_forward(self, X):
        for layer in self._layers:
            X = layer.activate(X)

        for i in range(X.shape[0]):
            if (X[i, :][0] > 0.5):
                X[i, :][0] = 1
            else:
                X[i, :][0] = 0
            if (X[i, :][1] > 0.5):
                X[i, :][1] = 1
            else:
                X[i, :][1] = 0

        return X

    def output(self, X):
        y = np.zeros((X.shape[0], 2), dtype=int)
        for i in range(X.shape[0]):
            if (X[i, :][0] == 0):
                y[i, :][0] = X[i, :][0]
                y[i, :][1] = X[i, :][1]
            if (X[i, :][0] == 1):
                y[i, :][0] = X[i, :][0]
                y[i, :][1] = X[i, :][2]

        return y

    def plot_z(self, X):
        x_predict = self.feed_forward(X)
        x_predict = self.output(x_predict)
        z = np.zeros((81, 81))
        for i in range(x_predict.shape[0]):
            if (x_predict[i] == [0, 0]).all():
                z[divmod(i, 81)[0], divmod(i, 81)[1]] = 139
            elif (x_predict[i] == [1, 1]).all():
                z[divmod(i, 81)[0], divmod(i, 81)[1]] = 255
            elif (x_predict[i] == [0, 1]).all():
                z[divmod(i, 81)[0], divmod(i, 81)[1]] = 0
            elif (x_predict[i] == [1, 0]).all():
                z[divmod(i, 81)[0], divmod(i, 81)[1]] = 0
        return z.T


if __name__ == '__main__':
    # create data
    x, y = np.mgrid[-10:10.25:0.25, -10:10.25:0.25]
    x_test = np.c_[np.ravel(x), np.ravel(y)]
    y_test = np.array(x_test)
    for i in range(x_test.shape[0]):
        if (x_test[i][0] <= 0):
            y_test[i] = [0, 1]
        if (x_test[i][0] > 0):
            y_test[i] = [1, 0]
        if (x_test[i][0] <= -7 and x_test[i][0] >= -9 and x_test[i][1] >= -5 and x_test[i][1] <= 5):
            y_test[i] = [0, 0]
        if (x_test[i][0] <= -1 and x_test[i][0] >= -3 and x_test[i][1] >= -5 and x_test[i][1] <= 5):
            y_test[i] = [0, 0]
        if (x_test[i][0] <= -3 and x_test[i][0] >= -7 and x_test[i][1] >= -5 and x_test[i][1] <= -2):
            y_test[i] = [0, 0]
        if (x_test[i][0] <= 4 and x_test[i][0] >= 2 and x_test[i][1] >= -5 and x_test[i][1] <= 5):
            y_test[i] = [1, 1]
        if (x_test[i][0] <= 8 and x_test[i][0] >= 4 and x_test[i][1] >= 3 and x_test[i][1] <= 5):
            y_test[i] = [1, 1]
        if (x_test[i][0] <= 6 and x_test[i][0] >= 4 and x_test[i][1] >= -1 and x_test[i][1] <= 1):
            y_test[i] = [1, 1]

    weights1 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])

    bias1 = np.array([10, 9, 7, 3, 1, 0, -2, -4, -6, -8, -10, -10, -5, -3, -1, 1, 2, 5, 10])
    weigths2 = np.array([[-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
                         [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0],
                         [0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0],
                         [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0],
                         [0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
                         [0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0],
                         [0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0],
                         [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
                         [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 1, -1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0],
                         [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0],
                         [0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 1, -1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 1, -1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 1, -1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 1, 0, -1, 0],
                         [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, -1]])

    bias2 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    weigths3 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         [0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 0, 0, 0]])
    bias3 = np.array([-8, -10, -3, 2])

    weigths4 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 0, 1]])
    bias4 = np.array([-1, -1 / 2, -2])
    nn = NeuralNetwork()
    nn.add_layer(Layer(2, 19, weights=weights1, bias=bias1, activation='step'))
    nn.add_layer(Layer(19, 18, weights=weigths2.T, bias=bias2, activation='step'))
    nn.add_layer(Layer(18, 4, weights=weigths3.T, bias=bias3, activation='step'))
    nn.add_layer(Layer(4, 3, weights=weigths4.T, bias=bias4, activation='step'))

    z = nn.plot_z(x_test)
    plt.imshow(z)
    a = plt.xticks(np.arange(0, 81, 1), np.arange(-10, 10.25, 0.25), rotation=90)
    plt.yticks(np.arange(0, 81, 1), np.arange(-10, 10.25, 0.25))
    plt.grid()
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(4))
    ax.invert_yaxis()
    plt.savefig('./uf.png')
    plt.show()

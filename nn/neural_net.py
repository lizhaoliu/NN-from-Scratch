import numpy as np


class NeuralNet:
    """A neural network model.
    """

    def __init__(self, neurons_per_layer, activation_fn='relu', loss_fn='square_loss'):
        """

        :param neurons_per_layer: A list of integers that denote the number of neurons in each layer.
        """
        if not isinstance(neurons_per_layer, list):
            raise TypeError("neurons_per_layer must be a list.")
        num_layers = len(neurons_per_layer)
        if num_layers < 2:
            raise ValueError("There must be at least 2 layers (an input and an output layers).")

        if activation_fn == 'relu':
            self.__activation_fn = _relu
            self.__activation_fn_diff = _relu_diff
        elif activation_fn == 'sigmoid':
            self.__activation_fn = _sigmoid
            self.__activation_fn_diff = _sigmoid_diff
        else:
            raise ValueError("Unknown activation function.")

        if loss_fn == 'square_loss':
            self.__loss_fn = _square_loss
            self.__loss_fn_diff = _square_loss_diff
        else:
            raise ValueError("Unknown loss function.")

        self.__activation_fn = activation_fn
        self.__loss_fn = loss_fn

        self.__num_layers = num_layers
        self.__w = [0]  # Dummy for layer 0 (input layer).
        self.__b = [0]  # Dummy for layer 0 (input layer).
        for i in range(1, num_layers):
            w = np.random.standard_normal(size=(neurons_per_layer[i - 1], neurons_per_layer[i]))
            self.__w.append(w)
            b = np.random.standard_normal(size=neurons_per_layer[i])
            self.__b.append(b)

    def __ff(self, x):
        """Calculate the feed-forward result.

        :param x: The input data, in shape [batch_size, num_features]. Note: num_features must match the number of
        neurons in the first layer.
        :return: 
        """
        if np.ndim(x) < 2:
            x = np.reshape(x, [1, -1])
        if np.ndim(x) > 2:
            raise ValueError("ndim(x) > 2.")
        shape = np.shape(x)
        if shape[1] != self.__w[0].shape[0]:
            raise ValueError("x and the number of first layer do not match.")

        z = [x]
        a = [x]
        for i in range(1, self.__num_layers):
            z.append(np.matmul(x, self.__w[i]) + self.__b[i])
            a.append(self.__activation_fn(z[-1]))

        return z, a

    def bp(self, x, y, learning_rate):
        """Run back-propagation once.

        :param x: A batch of training data, in shape [batch_size, num_features].
        :param y: A batch of ground truth labels, in shape [batch_size].
        :param learning_rate: Learning rate.
        :return:
        """
        z, a = self.__ff(x)
        y_ff = a[-1]
        loss = self.__loss_fn(y_ff, y)
        print('Loss: {:.4f}.'.format(loss))

        # Calculate deltas.
        deltas = np.zeros([self.__num_layers])
        deltas[-1] = self.__loss_fn_diff(a[-1], y) * self.__activation_fn_diff(z[-1])
        for l in range(self.__num_layers - 2, 0, -1):
            deltas[l] = np.matmul(self.__w[l + 1], np.reshape(deltas[l + 1]), [-1, 1]) * self.__activation_fn_diff(z[l])

        # Calculate gradient and update.
        for l in range(self.__num_layers - 1, 0, -1):
            self.__w[l] -= learning_rate * np.matmul(np.reshape(a[l - 1], [-1, 1]),
                                                     np.reshape(deltas[l], [1, -1]))
            self.__b[l] -= learning_rate * deltas[l]


def _relu(x):
    zeros = np.zeros_like(x)
    return np.maximum(zeros, x)


def _relu_diff(x):
    x[x > 0] = 1.0
    x[x < 0] = 0.0
    return x


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _sigmoid_diff(x):
    x = np.exp(-x)
    return x / np.power(1 + x, 2)


def _square_loss(x, y):
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape.")
    d = np.power(y - x, 2) / 2
    return np.sum(d)


def _square_loss_diff(x, y):
    return x - y

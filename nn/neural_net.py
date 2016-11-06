import numpy as np


class NeuralNet:
    """A neural network model.
    """

    def __init__(self, neurons_per_layer):
        """

        :param neurons_per_layer: A list of integers that denote the number of neurons in each layer.
        """
        if not isinstance(neurons_per_layer, list):
            raise TypeError("neurons_per_layer must be a list.")
        num_layers = len(neurons_per_layer)
        if num_layers < 2:
            raise ValueError("There must be at least 2 layers (an input and an output layers).")

        self.__num_layers = num_layers
        self.__w = []
        self.__b = []
        for i in range(1, num_layers):
            w = np.random.standard_normal(size=(neurons_per_layer[i - 1], neurons_per_layer[i]))
            self.__w.append(w)
            b = np.random.standard_normal(size=neurons_per_layer[i])
            self.__b.append(b)

    def ff(self, x, activate_fn=relu):
        """Calculate the feed-forward result.
        :param x: The input data, in shape [batch_size, num_features]. Note: num_features must match the number of
        neurons in the first layer.
        :param activate_fn: The activate function.
        :return: The
        """
        if np.ndim(x) < 2:
            x = np.reshape(x, [1, -1])
        if np.ndim(x) > 2:
            raise ValueError("ndim(x) > 2.")
        shape = np.shape(x)
        if shape[1] != self.__w[0].shape[0]:
            raise ValueError("x and the number of first layer do not match.")

        for i in range(len(self.__w)):
            z = np.matmul(x, self.__w[i]) + self.__b[i]
            x = activate_fn(z)

        return x

    def bp(self, x, y, learning_rate):
        """Run back propagation once.
        :param x:
        :param y:
        :param learning_rate:
        :return:
        """
        pass


def relu(x):
    """

    :param x:
    :return:
    """
    zeros = np.zeros_like(x)
    return np.maximum(zeros, x)


def sigmoid(x):
    """

    :param x:
    :return:
    """
    return 1 / (1 + np.log(-x))


def tanh(x):
    """

    :param x:
    :return:
    """
    return np.tanh(x)


def softplus(x):
    """

    :param x:
    :return:
    """
    return np.log(np.exp(x) + 1)

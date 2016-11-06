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
        elif activation_fn == 'softplus':
            self.__activation_fn = _softplus
            self.__activation_fn_diff = _softplus_diff
        else:
            raise ValueError("Unknown activation function.")

        if loss_fn == 'square_loss':
            self.__loss_fn = _square_loss
            self.__loss_fn_diff = _square_loss_diff
        else:
            raise ValueError("Unknown loss function.")

        self.__num_layers = num_layers
        self.__w = [None]  # Dummy for layer 0 (input layer).
        self.__b = [None]  # Dummy for layer 0 (input layer).
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
            raise ValueError("Dimension of x must not be greater than 2.")
        if np.shape(x)[1] != self.__w[1].shape[0]:
            raise ValueError("x and the number of first layer do not match.")

        z, a = [x], [x]  # Input layer.
        for l in range(1, self.__num_layers):
            new_z = np.matmul(a[l - 1], self.__w[l]) + self.__b[l]
            z.append(new_z)
            a.append(self.__activation_fn(new_z))

        return z, a

    def bp(self, x, y, learning_rate):
        """Run back-propagation once.

        :param x: A batch of training data, in shape [batch_size, num_features].
        :param y: A batch of ground truth labels, in shape [batch_size].
        :param learning_rate: Learning rate.
        :return:
        """
        batch_size = x.shape[0]

        z, a = self.__ff(x)
        y_ff = a[-1]
        loss = self.__loss_fn(y_ff, y)
        print('Loss: {:.4f}.'.format(loss))

        # Calculate deltas.
        deltas = [None] * self.__num_layers
        deltas[-1] = self.__loss_fn_diff(a[-1], y) * self.__activation_fn_diff(z[-1])
        for l in range(self.__num_layers - 2, 0, -1):
            deltas[l] = np.matmul(deltas[l + 1], np.transpose(self.__w[l + 1])) * self.__activation_fn_diff(z[l])

        # Calculate gradient and update.
        for l in range(self.__num_layers - 1, 0, -1):
            gradient_w = np.matmul(np.transpose(a[l - 1]), deltas[l]) / batch_size
            self.__w[l] -= learning_rate * gradient_w
            gradient_b = np.mean(deltas[l], axis=0)  # Reduce mean on the batch size.
            self.__b[l] -= learning_rate * gradient_b
            print('Gradient[{}] -- w: {}, b: {}'.format(l, gradient_w, gradient_b))

    def ff(self, x):
        """Calculate the feed-forward result of a given input.

        :param x: A batch of input, in shape [batch_size, num_features].
        :return:
        """
        _, a = self.__ff(x)
        return a[-1]


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


def _softplus(x):
    return np.log(np.exp(x) + 1)


def _softplus_diff(x):
    x = np.exp(x)
    return x / (x + 1)


def _square_loss(x, y):
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape.")
    d = np.power(y - x, 2) / 2
    return np.sum(d)


def _square_loss_diff(x, y):
    return x - y

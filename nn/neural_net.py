import numpy as np


class NeuralNet:
    """A neural network model.
    """

    def __init__(self, neurons_per_layer, activation_fn='relu', loss_fn='square_loss', weight_decay=0.0):
        """Construct a neural network object.

        :param neurons_per_layer: A list of integers that denote the number of neurons in each layer.
        :param activation_fn: The activation function to use.
        :param loss_fn: The loss function to calculate loss.
        :param weight_decay: Weight decay for L2 regularization, default to 0 (no regularization).
        """
        # Validate layer information.
        if not isinstance(neurons_per_layer, list):
            raise TypeError('neurons_per_layer must be a list.')
        num_layers = len(neurons_per_layer)
        if num_layers < 2:
            raise ValueError('There must be at least 2 layers (an input and an output layers).')
        self.__num_layers = num_layers

        # Validate and set activation function.
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
            raise ValueError('Unknown activation function: {}.'.format(activation_fn))

        # Validate and set loss function.
        if loss_fn == 'square_loss':
            self.__loss_fn = _square_loss
            self.__loss_fn_diff = _square_loss_diff
        else:
            raise ValueError("Unknown loss function: {}.".format(loss_fn))

        # Validate and set weight decay.
        if weight_decay < 0:
            raise ValueError('Weight decay must be non-negative: {}.'.format(weight_decay))
        self.__wd = weight_decay

        # Initialize weights and biases for all layers.
        self.__w, self.__b = [0], [0]  # Dummy for layer 0 (input layer).
        for i in range(1, num_layers):
            # Initialize weights for layer i, with normal random variables.
            w = np.random.standard_normal(size=(neurons_per_layer[i - 1], neurons_per_layer[i]))
            self.__w.append(w)

            # Initialize the biases.
            b = np.zeros(neurons_per_layer[i])
            # b = np.random.standard_normal(size=neurons_per_layer[i])
            self.__b.append(b)

    def __ff(self, x):
        """Calculate the feed-forward result of each neuron (pre and post activation values) in the network.

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
            # z[l] = a[l - 1] * W[l] + b[l].
            new_z = np.matmul(a[l - 1], self.__w[l]) + self.__b[l]
            # a[l] = f(z[l]).
            new_a = self.__activation_fn(new_z)

            z.append(new_z)
            a.append(new_a)

        return z, a

    def bp(self, x, y, learning_rate):
        """Run back-propagation once.

        :param x: A batch of training data, in shape [batch_size, num_features].
        :param y: A batch of ground truth labels, in shape [batch_size].
        :param learning_rate: Learning rate.
        :return: The loss value before back-propagation updates the weights and biases.
        """
        z, a = self.__ff(x)
        y_ff = a[-1]
        loss = self.__loss_fn(y_ff, y)

        # Calculate deltas.
        deltas = [None] * self.__num_layers
        deltas[-1] = self.__loss_fn_diff(a[-1], y) * self.__activation_fn_diff(z[-1])
        for l in range(self.__num_layers - 2, 0, -1):
            deltas[l] = np.matmul(deltas[l + 1], np.transpose(self.__w[l + 1])) * self.__activation_fn_diff(z[l])

        # Calculate gradient and update.
        for l in range(self.__num_layers - 1, 0, -1):
            # d(C)/d(W[l]) = a[l - 1] * delta[l] + wd * W[l]. Must divided by batch_size.
            gradient_w = np.matmul(np.transpose(a[l - 1]), deltas[l]) / x.shape[0] + self.__wd * self.__w[l]
            self.__w[l] -= learning_rate * gradient_w

            # d(C)/d(b[l]) = delta[l]. Must divided by batch_size.
            gradient_b = np.mean(deltas[l], axis=0)  # Reduce mean on the batch size.
            self.__b[l] -= learning_rate * gradient_b
        return loss

    def ff(self, x):
        """Calculate the feed-forward result for a given input (the last layer's output).

        :param x: A batch of input, in shape [batch_size, num_features].
        :return: The predicted
        """
        _, a = self.__ff(x)
        return a[-1]


def _relu(x):
    """Returns the ReLU of the input data, i.e. max(0, x).

    :param x: A numpy array or scalar.
    :return:
    """
    zeros = np.zeros_like(x)
    return np.maximum(zeros, x)


def _relu_diff(x):
    """Returns the differential value of the input data.

    :param x: A numpy array or scalar.
    :return:
    """
    x[x > 0] = 1.0
    x[x < 0] = 0.0
    return x


def _sigmoid(x):
    """Returns the sigmoid of the input data, i.e. 1 / (1 + e^(-x)).

    :param x: A numpy array or scalar.
    :return:
    """
    return 1 / (1 + np.exp(-x))


def _sigmoid_diff(x):
    """Return the differential function value of sigmoid.

    :param x: A numpy array or scalar.
    :return:
    """
    x = np.exp(-x)
    return x / np.power(1 + x, 2)


def _softplus(x):
    """Returns the softplus of input data, i.e. log(1 + e^(x)).

    :param x: A numpy array or scalar.
    :return:
    """
    return np.log(np.exp(x) + 1)


def _softplus_diff(x):
    """Returns the differential function value of softplus on x.

    :param x: A numpy array or scalar.
    :return:
    """
    x = np.exp(x)
    return x / (x + 1)


def _square_loss(y_pred, y_label):
    """Returns the square loss of predicted values and ground truth labels.

    :param y_pred: The predicted y values, in shape [batch_size, 1].
    :param y_label: The ground truth y values, in shape [batch_size, 1].
    :return: The total loss.
    """
    if y_pred.shape != y_label.shape:
        raise ValueError("x and y must have same shape.")
    d = np.power(y_label - y_pred, 2) / 2
    return np.sum(d)


def _square_loss_diff(y_pred, y_label):
    """Returns the differential function value (on y_pred) of predicted values and ground truth labels.

    :param y_pred: Predicted y values.
    :param y_label: The ground truth y values.
    :return:
    """
    return y_pred - y_label


def _cross_entropy(y_pred, y_label):
    """Returns the cross entropy of predicted labels and ground truth labels.

    :param y_pred: Predicted probabilities of each class, in shape [batch_size, num_classes].
    :param y_label: One-hot encoded label values, in shape [batch_size, num_classes].
    :return: The total cross entropy loss.
    """
    if y_pred.shape != y_label.shape:
        raise ValueError("y_pred and y_label must have the same shape.")
    return np.mean(np.sum(np.log(y_pred) * y_label, axis=1))


def _cross_entropy_diff(y_pred, y_label):
    """Returns the differential function value (on y_pred) of predicted values and ground truth labels.

    :param y_pred: Predicted probabilities of each class, in shape [batch_size, num_classes].
    :param y_label: One-hot encoded label values, in shape [batch_size, num_classes].
    :return:
    """
    pass

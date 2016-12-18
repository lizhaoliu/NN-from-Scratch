import numpy as np


class NeuralNet:
    """A neural network model.
    """

    def __init__(self, neurons_per_layer, activation_fn='relu', output_fn=None, loss_fn='square_loss',
                 weight_decay=0.0):
        """Construct a neural network object.

        :param neurons_per_layer: A list/tuple of integers that denote the number of neurons in each layer.
        :param activation_fn: Activation function, e.g. 'relu', 'sigmoid', etc.
        :param output_fn: Function to compute the output layer, e.g. 'softmax', 'sigmoid'. None means not to operator
        on the output logits.
        :param loss_fn: Loss function to calculate loss, can be 'cross_entropy_loss' or 'square_loss'.
        :param weight_decay: Weight decay for L2 regularization, default to 0 (no regularization).
        """
        # Validate layer information.
        if isinstance(neurons_per_layer, tuple):
            neurons_per_layer = list(neurons_per_layer)
        if not isinstance(neurons_per_layer, list):
            raise TypeError('neurons_per_layer must be a list.')
        num_layers = len(neurons_per_layer)
        if num_layers < 2:
            raise ValueError('There must be at least 2 layers (an input and an output layers).')
        self.__num_layers = num_layers

        # Validate and set activation function and it's derivative function.
        if activation_fn == 'relu':
            self.__activation_fn = _relu
            self.__activation_fn_deriv = _relu_deriv
        elif activation_fn == 'sigmoid':
            self.__activation_fn = _sigmoid
            self.__activation_fn_deriv = _sigmoid_deriv
        elif activation_fn == 'softplus':
            self.__activation_fn = _softplus
            self.__activation_fn_deriv = _softplus_deriv
        else:
            raise ValueError('Unknown activation function: {}.'.format(activation_fn))

        # Validate and set output function.
        if output_fn == 'softmax':
            self.__output_fn = _softmax
        else:
            raise ValueError('Unknown output function: {}'.format(output_fn))

        # Validate and set loss function.
        if loss_fn == 'square_loss':
            self.__loss_fn = _square_loss
            self.__loss_fn_deriv = _square_loss_deriv
        elif loss_fn == 'cross_entropy_loss':
            self.__loss_fn = _cross_entropy
            self.__loss_fn_deriv = _cross_entropy_deriv
        else:
            raise ValueError("Unknown loss function: {}.".format(loss_fn))

        # Validate and set weight decay.
        if weight_decay < 0:
            raise ValueError('Weight decay must be non-negative: {}.'.format(weight_decay))
        self.__wd = weight_decay

        # Initialize weights and biases for all layers.
        self.__w, self.__b = [0], [0]  # Placeholder values for layer 0 (input layer).
        for i in range(1, num_layers):
            # Initialize weights for layer i, with normal random variables, with mean = 0 and
            # stddev = sqrt(2 / (n[i - 1] + n[i])).
            w = np.random.normal(loc=0.0, scale=np.sqrt(2.0 / (neurons_per_layer[i - 1] + neurons_per_layer[i])),
                                 size=(neurons_per_layer[i - 1], neurons_per_layer[i]))
            self.__w.append(w)

            # Initialize the biases.
            b = np.zeros(neurons_per_layer[i])
            # b = np.random.standard_normal(size=neurons_per_layer[i])
            self.__b.append(b)

    def __feed_forward(self, x):
        """Calculate the feed-forward result of each neuron (pre and post activation values) in the network.

        :param x: Input data, in shape of [batch_size, num_features]. Note: num_features must match the number of
        neurons in the first layer.
        :return:
        """
        if np.ndim(x) < 2:
            x = np.reshape(x, [1, -1])
        if np.ndim(x) > 2:
            raise ValueError("Dimension of x must not be greater than 2.")
        if np.shape(x)[1] != self.__w[1].shape[0]:
            raise ValueError("Number of features and number neurons in first layer do not match.")

        # Track each neuron's pre and post activation values for back propagation.
        z, a = [x], [x]  # Input layer.
        for l in range(1, self.__num_layers):
            # z[l] = a[l - 1] * W[l] + b[l].
            new_z = np.matmul(a[l - 1], self.__w[l]) + self.__b[l]
            z.append(new_z)

            # If output layer, a = output(z).
            if l == self.__num_layers - 1:
                new_a = self.__output_fn(new_z)
            # otherwise a = activation(z).
            else:
                new_a = self.__activation_fn(new_z)
            a.append(new_a)

        return z, a

    def back_prop(self, x, y, learning_rate):
        """Run back-propagation and update parameters for one time.

        :param x: A batch of training data, in shape [batch_size, num_features].
        :param y: A batch of ground truth labels, in shape [batch_size].
        :param learning_rate: Learning rate.
        :return: The loss value before back-propagation updates the weights and biases.
        """
        z, a = self.__feed_forward(x)
        y_out = a[-1]
        loss = self.__loss_fn(y_out, y)

        # Calculate deltas, delta[l, j] = d(C)/d(z[l, j]), where C is the total cost/loss.
        deltas = [None] * self.__num_layers
        # Calculate output layer's delta values.
        # TODO: Implement this in a more generic way.
        deltas[-1] = self.__loss_fn_deriv(a[-1], y) * self.__activation_fn_deriv(z[-1])
        # Calculate other layers' delta values.
        for l in range(self.__num_layers - 2, 0, -1):
            deltas[l] = np.matmul(deltas[l + 1], np.transpose(self.__w[l + 1])) * self.__activation_fn_deriv(z[l])

        # Calculate gradient and update.
        for l in range(self.__num_layers - 1, 0, -1):
            # d(C)/d(W[l]) = a[l - 1] * delta[l] + wd * W[l]. Must divided by batch_size.
            gradient_w = np.matmul(np.transpose(a[l - 1]), deltas[l]) / x.shape[0] + self.__wd * self.__w[l]
            self.__w[l] -= learning_rate * gradient_w

            # d(C)/d(b[l]) = delta[l]. Must divided by batch_size.
            gradient_b = np.mean(deltas[l], axis=0)  # Reduce mean on the batch size.
            self.__b[l] -= learning_rate * gradient_b
        return loss

    def feed_forward(self, x):
        """Calculate the feed-forward result for a given input (the last layer's output).

        :param x: A batch of input, in shape [batch_size, num_features].
        :return: The predicted
        """
        _, a = self.__feed_forward(x)
        return a[-1]


def _relu(x):
    """Returns the ReLU of the input data, i.e. max(0, x).

    :param x: A numpy array or scalar.
    :return:
    """
    zeros = np.zeros_like(x)
    return np.maximum(zeros, x)


def _relu_deriv(x):
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


def _sigmoid_deriv(x):
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


def _softplus_deriv(x):
    """Returns the differential function value of softplus on x.

    :param x: A numpy array or scalar.
    :return:
    """
    x = np.exp(x)
    return x / (x + 1)


def _softmax(x):
    """Calculate the softmax values of given logits.

    :param x: Raw logits, in shape of [batch_size, num_classes].
    :return:
    """
    exp = np.exp(x)
    return exp / (np.sum(exp, axis=1)[:, np.newaxis])


def _square_loss(y_out, y_truth):
    """Returns the square loss of predicted values and ground truth labels.

    :param y_out: The output values, in shape [batch_size, 1].
    :param y_truth: The ground truth y values, in shape [batch_size, 1].
    :return: The total loss.
    """
    if y_out.shape != y_truth.shape:
        raise ValueError("x and y must have same shape.")
    d = np.power(y_truth - y_out, 2) / 2
    return np.sum(d)


def _square_loss_deriv(y_out, y_truth):
    """Returns the differential function value (on y_pred) of predicted values and ground truth labels.

    :param y_out: Predicted y values.
    :param y_truth: The ground truth y values.
    :return:
    """
    return y_out - y_truth


def _cross_entropy(y_out, y_label):
    """Returns the cross entropy of predicted labels and ground truth labels.

    :param y_out: Predicted probabilities of each class, in shape [batch_size, num_classes].
    :param y_label: One-hot encoded label values, in shape [batch_size, num_classes].
    :return: The total cross entropy loss.
    """
    if y_out.shape != y_label.shape:
        raise ValueError("y_pred and y_label must have the same shape.")
    return np.mean(np.sum(-np.log(y_out) * y_label, axis=1))


def _cross_entropy_deriv(y_out, y_label):
    """Returns the differential function value (on y_pred) of predicted values and ground truth labels.

    :param y_out: Predicted probabilities of each class, in shape [batch_size, num_classes].
    :param y_label: One-hot encoded label values, in shape [batch_size, num_classes].
    :return:
    """
    if y_out.shape != y_label.shape:
        raise ValueError("y_out and y_label must have the same shape.")
    return np.mean(np.sum(-y_label / y_out, axis=1))

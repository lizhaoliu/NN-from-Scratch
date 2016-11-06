import numpy as np
import pandas as pd
from sklearn import model_selection

from nn import neural_net

DATA_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

df = pd.read_csv(DATA_URL, sep=',', header=None)
one_hot = np.reshape(pd.Categorical(df[4]).codes, [-1, 1])
# one_hot = pd.get_dummies(df[4])
df = df.drop(4, axis=1)
data = np.concatenate([df.values, one_hot], axis=1)
np.random.shuffle(data)
x, y = data[:, :-1], data[:, -1:]
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.75)

net = neural_net.NeuralNet([x_train.shape[-1], 6, 6, y_train.shape[-1]], activation_fn='softplus')
for iter in range(10000):
    net.bp(x_train, y_train, 0.1)

y_ff = net.ff(x_test)
print(y_ff, y_test)
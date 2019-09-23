# Models for Miniproject 1
import numpy as np
import pandas as pd
import utils


class model(object):
    def __init__(self):
        return

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def evaluate_acc(self, X, y):
        return sum(self.predict(X) == y) / y.size  # in [0,1]


class logistic_regression(model):
    alpha_init = 10
    num_epoch = 20

    def __init__(self, m, alpha_mode='hyperbolic', **kwargs):
        # m : number of features (dimensions) of the linear model

        self.w = np.random.randn(m)
        self.update_alpha = switch_update_alpha[alpha_mode]

    def fit(self, X_train, y_train, X_val, y_val):
        alpha = self.alpha_init
        metrics = np.zeros((self.num_epoch, 2))
        for epoch in range(self.num_epoch):
            dEdw = 0
            E = 0
            for isample in range(y_train.size):
                x_i = X_train[isample]
                y_i = y_train[isample]
                a = np.dot(self.w, x_i)
                dEdw += x_i * (y_i - sigma(a))

                # loading bar
                utils.printProgressBar(isample, y_train.size, prefix="epoch {}/{}".format(epoch, self.num_epoch),
                                       length=20)

            # gradient descent step
            self.w = self.w + alpha * dEdw
            # update step
            alpha = self.update_alpha(alpha, epoch)

            # metrics
            metrics[epoch, :2] = np.array([self.evaluate_acc(X_train, y_train), self.evaluate_acc(X_val, y_val)])
            print("train acc. {} val acc. {}\n".format(metrics[epoch, 0], metrics[epoch, 1]))

        return metrics

    def predict(self, X):
        return (sigma(np.dot(X, self.w)) > 0.5).astype('int64')


# sigmoid function
def sigma(x):
    return 1 / (1 + np.exp(-x))


# alpha methods
switch_update_alpha = {'constant': (lambda alpha, k: alpha),
                       'hyperbolic': (lambda alpha, k: alpha * (k + 1) / (k + 2))
                       }

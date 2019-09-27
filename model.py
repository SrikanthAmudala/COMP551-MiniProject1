# Models for Miniproject 1
import numpy as np
import pandas as pd
import utils
from matplotlib import pyplot as plt


class model(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return 0

    @staticmethod
    def evaluate_acc(y_pred, y_true):
        return sum(y_pred == y_true) / y_true.size  # in [0,1]

    def do_metrics(self, X_train, y_train, X_val, y_val):
        y_pred_train = self.predict(X_train)
        y_pred_val = self.predict(X_val)
        return [self.evaluate_acc(y_pred=y_pred_train,
                                  y_true=y_train),
                self.evaluate_acc(y_pred=y_pred_val,
                                  y_true=y_val)]


class logistic_regression(model):
    # alpha methods
    @staticmethod
    def _update_alpha_constant(alpha, k):
        return alpha

    def _update_alpha_hyperbolic(self, alpha, k):
        return alpha * (k) / (self.decay * (k + self.beta))

    def _update_alpha_mixed(self,alpha,k):
        return alpha / 10 if np.abs(np.log10(1/alpha) - 2 - k/100) < 1e-9 else alpha

    # different stopping criterions :
    # epoch : fixed number of iterations
    # convergence : detect convergence, if not default epoch max number
    def _stopping_condition_epoch(self, epoch, **kwargs):
        return epoch != self.num_epoch

    def _stopping_condition_convergence(self, epoch, delta, **kwargs):
        return (np.abs(delta >= self.threshold)) and (epoch != self.num_epoch)

    def _regularization_none(self):
        return 0
    
    def _regularization_l2(self):
        return (- self.lam * self.w)

    def _regularization_l1(self):
        return (- self.lam * np.sign(self.w))

    def __init__(self, m,
                 alpha_mode='hyperbolic', alpha_init=1, decay=1, beta = 1,
                 stopping_mode='convergence', num_epoch=20, threshold=1,
                 train_metrics=False,
                 regularization_mode='none', lam = 0.1
                 ):
        # m : number of features (dimensions) of the linear model
        self.w = np.zeros(m)
        self.alpha_init = alpha_init
        self.decay = decay
        self.beta = beta
        self.num_epoch = int(num_epoch)
        self.threshold = threshold
        self.train_metrics = train_metrics
        self.lam = lam

        self.update_alpha = getattr(self, '_update_alpha_' + alpha_mode)
        self.stopping_condition = getattr(self, '_stopping_condition_' + stopping_mode)
        self.regularization = getattr(self,'_regularization_' + regularization_mode)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # initial step size of the gradient descent
        alpha = self.alpha_init

        # loop control variables
        epoch = 0
        delta = self.threshold

        # record of the training and validation accuracy per epoch
        if self.train_metrics:
            metrics = np.zeros((self.num_epoch + 1, 2))
            metrics[epoch, :] = self.do_metrics(X_train, y_train, X_val, y_val)
            print("epoch {} : train acc. {} val acc. {}\n".format(epoch, metrics[epoch, 0], metrics[epoch, 1]))

        while self.stopping_condition(epoch=epoch, delta=np.linalg.norm(delta)):
            epoch += 1
            dEdw = 0
            for isample in range(y_train.size):
                x_i = X_train[isample]
                y_i = y_train[isample]
                a = np.dot(self.w, x_i)
                dEdw += x_i * (y_i - sigma(a))

            # gradient descent step
            delta = alpha * (dEdw + self.regularization())
            self.w += delta

            # update step
            alpha = self.update_alpha(alpha, epoch)

            # metrics
            if self.train_metrics:
                metrics[epoch, :] = self.do_metrics(X_train, y_train, X_val, y_val)
                print("epoch {} : train acc. {} val acc. {}\n".format(epoch, metrics[epoch, 0], metrics[epoch, 1]))

        # metrics
        if self.train_metrics:
            if epoch != self.num_epoch:
                metrics[epoch:, 0].fill(metrics[epoch-1,0])
                metrics[epoch:, 1].fill(metrics[epoch-1,1])
            
            return metrics

    def predict(self, X):
        return (sigma(np.dot(X, self.w)) > 0.5).astype('int64')

class lda(model):
    def __init__(self,m):
        self.train_metrics = False
        self.w = np.zeros(m)

    def fit(self, X_train, y_train):
        if X_train.shape[1] == self.w.size:
            X_train = X_train[:,1:] # remove bias term !

        N0 = y_train[y_train == 0].size
        N1 = y_train[y_train == 1].size

        # p(y=0), p(y=1)
        p0 = N0 / y_train.shape[0]
        p1 = N1 / y_train.shape[0]

        X_train_0 = X_train[y_train == 0]
        X_train_1 = X_train[y_train == 1]

        mean = [np.mean(X_train_0,axis=0),
                np.mean(X_train_1,axis=0)]

        # covar
        x_mu_0 = X_train_0 - mean[0]
        x_mu_1 = X_train_1 - mean[1]
        covar = (x_mu_0.T @ x_mu_0  + x_mu_1.T @ x_mu_1) / (N0 + N1 - 2)
        invcovar = np.linalg.inv(covar)

        self.w[0] = np.log(p1 / p0) + 0.5 * ((mean[0] @ (invcovar @ mean[0])) - (mean[1] @ (invcovar @ mean[1]))) 

        self.w[1:] = invcovar @ (mean[1] - mean[0])

    def predict(self, X):
        return ((np.dot(X, self.w)) > 0).astype('int64')

## Package
def kfold(model_init, df, k=5, **params):
    # init
    dataset = df.to_numpy()
    np.random.shuffle(dataset)
    mymodel = model_init(dataset.shape[1], **params) # remove 1 for target but add 1 for bias

    # metrics
    if mymodel.train_metrics:
        metrics = np.zeros((mymodel.num_epoch + 1, 2))
        # compute baseline for plot
        counts = df.iloc[:, -1].value_counts().sort_values(ascending=False)
        baseline = counts.iloc[0] / (counts.iloc[0] + counts.iloc[1])
    else:
        metrics = np.zeros((1, 2))

    # kfold loop
    folds = np.array_split(dataset, k, axis=0)
    for i in range(k):
        dataset_val = folds[i]
        dataset_train = np.vstack(folds[:i] + folds[i + 1:])
        (X_train, y_train, X_val, y_val) = utils.preprocessing_kfold(dataset_train, dataset_val)
        
        # metrics
        if mymodel.train_metrics:
            metrics += mymodel.fit(X_train, y_train, X_val, y_val)
        else:
            mymodel.fit(X_train, y_train)
            metrics += mymodel.do_metrics(X_train,y_train,X_val,y_val)

    metrics /= k
    
    # plot metrics
    if mymodel.train_metrics:
        plt.figure()
        plt.plot(metrics, '.-')
        plt.plot(np.array([0, metrics.shape[0]-1]), baseline * np.ones(2), '-r')
        plt.xlabel('epochs')
        plt.ylabel('metrics')
        plt.legend(['train acc.', 'val. acc.', 'baseline'])
        plt.grid()
        plt.show()

    return (metrics[-1, :],mymodel.w)

# sigmoid function
def sigma(x):
    return 1 / (1 + np.exp(-x))

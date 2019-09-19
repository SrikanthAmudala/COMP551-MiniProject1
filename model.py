# Models for Miniproject 1
import numpy as np
import pandas as pd
import utils

class model(object):
    def __init__(self):
        pass

    def fit(self,X,y):
        pass

    def predict(self,X):
        pass
    
    def evaluate_acc(self,X,y):
        return sum(self.predict(X) == y) / y.size # in [0,1]

class logistic_regression(model):
    # alpha methods
    @staticmethod
    def _update_alpha_constant(alpha,k): 
        return alpha

    @staticmethod
    def _update_alpha_hyperbolic(alpha,k): 
        return alpha * (k + 1) / (k + 2)

    # different stopping criterions :
    # epoch : fixed number of iterations
    # convergence : detect convergence, if not default epoch max number
    def _stopping_condition_epoch(self,epoch,**kwargs): 
        return epoch == self.num_epoch

    def _stopping_condition_convergence(self,epoch,delta,**kwargs):
        if (np.abs(delta < self.threshold)):
            print("converged\n")
            return False
        if  (epoch == self.num_epoch):
            return False
        
        return True


    def __init__(self, m, 
                 alpha_mode = 'hyperbolic', alpha_init = 1,
                 stopping_mode = 'convergence', num_epoch = 20, threshold = 1
                 ):
        # m : number of features (dimensions) of the linear model
        self.w = np.random.randn(m)
        self.alpha_init = alpha_init
        self.num_epoch = int(num_epoch)
        self.threshold = threshold

        self.update_alpha = getattr(self,'_update_alpha_' + alpha_mode)
        self.stopping_condition = getattr(self,'_stopping_condition_' + stopping_mode)

    def fit(self,X_train,y_train,X_val,y_val):
        # initial step size of the gradient descent
        
        alpha = self.alpha_init

        # record of the training and validation accuracy per epoch
        metrics = np.zeros((self.num_epoch,2))
        
        # loop control variables
        epoch = 0
        delta = self.threshold
        while self.stopping_condition(epoch = epoch, delta = np.linalg.norm(delta)):
            dEdw = 0
            for isample in range(y_train.size):
                x_i = X_train[isample]
                y_i = y_train[isample]
                a = np.dot(self.w,x_i)
                dEdw += x_i * (y_i - sigma(a))

            # gradient descent step
            delta = alpha * dEdw
            self.w += delta

            # update step
            alpha = self.update_alpha(alpha,epoch)

            # metrics
            metrics[epoch,:2] = [self.evaluate_acc(X_train,y_train), self.evaluate_acc(X_val,y_val)]
            print("epoch {} : train acc. {} val acc. {}\n".format(epoch + 1, metrics[epoch,0], metrics[epoch,1]))

            epoch += 1
        
        return metrics[:epoch,:]

    def predict(self,X):
        return (sigma(np.dot(X,self.w)) > 0.5).astype('int64')

## Package
# sigmoid function
def sigma(x):
    return 1 / (1 + np.exp(-x))
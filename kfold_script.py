# test script

import numpy as np
import pandas as pd
import model
from matplotlib import pyplot as plt

# params kfold
k = 5
path = 'winequality/clean_redwine.csv'
# path = 'breastcancer/clean_breastcancer.csv'
num_epoch = 20
alpha_init = 1e-2
threshold = 1e-3
decay = 1.2
alpha_mode = 'hyperbolic'

train_metrics = False
stopping_mode = 'convergence'

df = pd.read_csv(path,index_col=0)
metrics = model.kfold(model.logistic_regression,
                df = df,
                k = k,
                num_epoch = num_epoch,
                alpha_init = alpha_init,
                threshold = threshold,
                decay = decay,
                alpha_mode = alpha_mode,
                train_metrics = train_metrics,
                stopping_mode = stopping_mode)

print("logistic model")
print("{}-fold cross-validation : train acc. {} val acc. {}\n".format(k, metrics[0], metrics[1]))

metrics2 = model.kfold(model.lda,
                df = df,
                k = k)

print("lda model")
print("{}-fold cross-validation : train acc. {} val acc. {}\n".format(k, metrics2[0], metrics2[1]))

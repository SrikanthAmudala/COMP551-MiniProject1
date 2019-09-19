# test script

import numpy as np
import pandas as pd
import utils
import model
from matplotlib import pyplot as plt

clean_redwine = pd.read_csv('winequality/clean_redwine.csv',index_col=0)

(X_train,y_train,X_val,y_val) = utils.preprocessing(clean_redwine,prop=0.8)

mymodel = model.logistic_regression(X_train.shape[1], num_epoch = 300, alpha_init = 0.1, threshold = 1e-6)
metrics = mymodel.fit(X_train,y_train,X_val,y_val)

# plot metrics
plt.figure()
plt.plot(metrics)
plt.xlabel('epochs')
plt.ylabel('metrics')
plt.legend(['train acc.','val. acc.'])
plt.show()

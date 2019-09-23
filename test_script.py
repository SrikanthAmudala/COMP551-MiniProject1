# test script

import numpy as np
import pandas as pd
import utils
import model
from matplotlib import pyplot as plt

clean_redwine = pd.read_csv('winequality/clean_redwine.csv',index_col=0)

counts = clean_redwine['quality'].value_counts().sort_values(ascending = False)
baseline = counts.iloc[0]/(counts.iloc[0] + counts.iloc[1])

(X_train,y_train,X_val,y_val) = utils.preprocessing(clean_redwine,prop=0.8)

mymodel = model.logistic_regression(X_train.shape[1], num_epoch = 1e3, alpha_init = 1e-3, threshold = 1e-6, decay = 1.2, alpha_mode='constant')
metrics = mymodel.fit(X_train,y_train,X_val,y_val)

# plot metrics
plt.figure()
plt.plot(metrics)
plt.plot(np.array([0,metrics.shape[0]]), baseline * np.ones(2),'-r')
plt.xlabel('epochs')
plt.ylabel('metrics')
plt.legend(['train acc.','val. acc.','baseline'])
plt.show()

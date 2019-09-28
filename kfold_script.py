# test script

import numpy as np
import pandas as pd
import model
import utils
import seaborn as sns
from matplotlib import pyplot as plt
plt.close('all')

# load optimized features by l1 regularization (lam = 10)
important_feature = np.load('important_feature.npy')
important_feature = important_feature[important_feature != 0] - 1

# params kfold
k = 5
path = 'winequality/clean_redwine.csv'
# path = 'breastcancer/clean_breastcancer.csv'
num_epoch_list = [500]
alpha_init_list = np.array([0.01])#np.linspace(0.005,0.05,11)
threshold = 1e-4
decay_list = [1]
beta_list = np.array([0.8])#np.linspace(0,2,11)
alpha_mode = 'hyperbolic'

train_metrics = True
stopping_mode = 'convergence'

regularization_mode = 'none'
lam = 5

df = pd.read_csv(path,index_col=0)
# df = utils.augment_interact(df).iloc[:,np.append(important_feature[:4],-1)]
# df = df.iloc[:,[10,-1]] # only alcohol
param_list = []
metric_list = []

print("logistic model")

# test different set of parameters
for alpha_init in alpha_init_list:
    for decay in decay_list:
        for beta in beta_list:
            for num_epoch in num_epoch_list:
                (metrics,w_log) = model.kfold(model.logistic_regression,
                                df = df,
                                k = k,
                                num_epoch = num_epoch,
                                alpha_init = alpha_init,
                                threshold = threshold,
                                decay = decay,
                                alpha_mode = alpha_mode,
                                train_metrics = train_metrics,
                                stopping_mode = stopping_mode,
                                regularization_mode = regularization_mode,
                                lam = lam)
                
                param_list.append([alpha_init,decay,beta,num_epoch])
                metric_list.append(metrics)
                print("{}-fold cross-validation : train acc. {} val acc. {}\n".format(k, metrics[0], metrics[1]))

print("lda model")

(metrics2,w_lda) = model.kfold(model.lda,
                 df = df,
                 k = k)

print("{}-fold cross-validation : train acc. {} val acc. {}\n".format(k, metrics2[0], metrics2[1]))

# heat map for parameter exploration
param_array = np.array(param_list)
metric_array = np.array(metric_list)
metric_df = pd.DataFrame(data=param_list,columns=['alpha_init','decay','beta','num_epoch'])
metric_df['train acc'] = metric_array[:,0]*100
metric_df['val acc'] = metric_array[:,1]*100
plt.figure()
sns.heatmap(data=metric_df.pivot(index='beta', columns='alpha_init', values='train acc'), 
            annot=True,cmap=plt.get_cmap('Reds'),
            xticklabels=np.round(alpha_init_list*1e4)/1e4,
            yticklabels=np.round(np.array(beta_list)*10)/10,linewidth=.5,fmt='1.2f')
plt.title('Training Accuracy [%]')

# weight vector barplot
plt.figure()
ax = sns.barplot(x=np.arange(w_log.size), y=w_log, palette="vlag")
plt.xlabel('feature')
ax.set_xticklabels([])
plt.ylabel('value')
plt.title('weight vector')
plt.grid()

# feature selection with L1 norm
#abs_wlog_lam10 = np.abs(w_log)
#important_feature = np.flip(np.argsort(abs_wlog_lam10))
#np.save('important_feature',important_feature)


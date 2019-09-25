import collections

import numpy as np
import pandas
import utils
from sklearn import metrics


class Lda:
    def __init__(self):
        self.mean = [0, 0]

    def fit(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        target_count = collections.Counter(y_train)

        # class zero

        self.N0 = target_count.get(0.0)
        self.N1 = target_count.get(1.0)

        # p(y=0), p(y=1)
        self.p0 = self.N0 / self.y_train.shape[0]
        self.p1 = self.N1 / self.y_train.shape[0]

        I = np.zeros([self.y_train.shape[0], 2])
        ctr = 0

        # mean
        for i, j in zip(self.y_train, self.X_train):
            if int(i) == 0:
                self.mean[0] += j
                I[ctr, 0] = 1
            else:
                self.mean[-1] += j
                I[ctr, 1] = 1
            ctr += 1

        self.mean = np.asarray(self.mean) / [[self.N0], [self.N1]]

        # covar
        x_mu_0 = X_train - self.mean[0]
        x_mu_1 = X_train - self.mean[1]

        cluster_0 = []
        cluster_1 = []

        for i in range(0, len(I)):
            cluster_0.append(I[i][0] * np.dot(x_mu_0[i].reshape(-1, 1), x_mu_0[i].reshape(-1, 1).T))
            cluster_1.append(I[i][1] * np.dot(x_mu_1[i].reshape(-1, 1), x_mu_1[i].reshape(-1, 1).T))

        cluster_0 = np.asarray(sum(cluster_0))
        cluster_1 = np.asarray(sum(cluster_1))

        self.covar = (cluster_0 + cluster_1) / (self.N0 + self.N1 - 2)

        self.w0 = np.log(self.p1 / self.p0) - 1 / 2 * np.dot(
            np.dot(self.mean[0].reshape(-1, 1).T, np.linalg.pinv(self.covar)),
            self.mean[0].reshape(-1, 1))
        self.w = np.dot(np.linalg.pinv(self.covar), self.mean[1] - self.mean[0])

    def predict(self, X_val):
        y_pred = ((self.w0 + np.dot(X_val, self.w)) > 0).astype('int64')
        return y_pred


obj = Lda()
df = pandas.read_csv("/Users/Srikanth/PycharmProjects/MiniProject1/breastcancer/clean_breastcancer.csv")
df = df.drop(['Sample code number'], axis=1)
X_train, y_train, X_val, y_val = utils.preprocessing(df)
obj.fit(X_train, y_train, X_val, y_val)
y_predit = obj.predict(X_val)

accuracy = metrics.accuracy_score(y_val.reshape(-1, 1), y_predit.reshape(-1, 1))
print(accuracy)

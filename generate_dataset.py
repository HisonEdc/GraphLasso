import celer
import numpy as np
import time
from sklearn.linear_model import Lasso
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error as mse

mse_200_sklearn = []
mse_200_celer = []
mse_1000_sklearn = []
mse_1000_celer = []
max_iter = [500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 50000, 70000, 100000]



# generate X, y & L matrix
n = 100
p = 1000
L = np.zeros((p, p))
for i in range(10):
    L[(i * p // 10), (i * p // 10):((i + 1) * (p // 10))] = -1
    L[(i * p // 10):((i + 1) * (p // 10)), (i * p // 10)] = -1
    L[i * p // 10, i * p // 10] = 0
for i in range(p):
    L[i, i] = -np.sum(L[i])

beta = np.zeros(p)
beta[:10] = 1
sigma = np.linalg.inv(L + np.diag([0.1]*p))
sigma_error = np.sqrt(beta @ sigma @ beta) / 2
X = np.random.multivariate_normal(mean=np.zeros(p), cov=sigma, size=n)
y = X @ beta + np.random.normal(loc=0, scale=sigma_error, size=n)

y = y - np.mean(y)
n = X.shape[0]
p = X.shape[1]
X = preprocessing.scale(X, axis=0)

lambda1 = 0.1
lambda2 = 0.08
lambdaL = 0.12

Lnew = lambdaL * L + lambda2 * np.eye(p)
eigL = np.linalg.eig(Lnew)
S = eigL[1].real @ np.sqrt(np.diag(eigL[0].real))
X_star = np.vstack((X, S.T)) / np.sqrt(2)
y_star = np.hstack((y, np.zeros(p)))

for iter in range(50):
    lasso = Lasso(alpha=0.1, max_iter=iter * 10, tol=1e-04)
    lasso.fit(X_star, y_star)
    y_pred = lasso.predict(X)
    mse_200_sklearn.append(mse(y, y_pred))

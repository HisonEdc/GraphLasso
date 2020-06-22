"""
    author: He Jiaxin
    time: 20/03/2020
    function: an implement of network-constrained regression
    version: v1.0
"""

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Lasso, LassoCV, ElasticNetCV
from graph_estimation import generate_graph
import graph_estimation
import time


def netreg(X, y, L=None, lambda1=0, lambda2=0, lambdaL=0, normalizeL=False):
    """
    netreg(X, y, L, lambda1, lambda2, lambdaL, normalizeL=False, K=10, verbose=False)

            Parameters
            ----------
            X:              2d array_like of numeric
                            standardized n (number of rows) by p (number of columns) design matrix.
            Y:              1d array_like of numeric
                            centered n by 1 vector of the response variable.
            L:              2d array_like of numeric
                            p by p symmetric matrix of the penalty weight matrix.
            lambda1:        float
                            tuning parameters of the lasso(norm-1) penalty.
            lambda2:        float
                            tuning parameters of the ridge(norm-2) penalty.
            lambdaL:        float
                            tuning parameters of the Laplacian matrix penalty.
            normalizeL:     boolean, True or False
                            binary parameter indicating whether the penalty weight matrix
                            needs to be normalized beforehand.
            K:              int
                            number of folds in cross-validation.
            verbose:        boolean, True or False
                            whether computation progress should be printed.

            Returns:
            ----------
            intercept:      intercept of the linear regression model.
            beta:           regression coefficients (slopes) of the linear regression model.
    """
    # t1 = time.perf_counter()
    _check_data(X, y, L)

    assert lambda1 >= 0 and lambda2 >= 0 and lambdaL >= 0, 'Penalty parameters must be non-negative.'
    assert not (lambda1 == 0 and lambda2 == 0 and lambdaL == 0), \
        'At least one of the tuning parameters must be positive.'
    # t2 = time.perf_counter()
    # print(t2 - t1)

    X_org = X
    y_org = y
    X_std = X.std(axis=0)
    # t3 = time.perf_counter()
    # print(t3 - t2)

    # data preprocessing
    y = y - np.mean(y)
    n = X.shape[0]
    p = X.shape[1]
    X = preprocessing.scale(X, axis=0)
    if L is not None and normalizeL:
        idx = np.where(np.diag(L) == 0)
        L[idx] = 1
        L = np.diag(1 / np.sqrt(np.diag(L))) @ L @ np.diag(1 / np.sqrt(np.diag(L)))
    elif L is None:
        L = graph_estimation.generate_graph(X, lambdaL, screen=True, symmetrize='and', threshold=1e-4, max_iter=10000)
        if normalizeL:
            idx = np.where(np.diag(L) == 0)
            L[idx] = 1
            L = np.diag(1 / np.sqrt(np.diag(L))) @ L @ np.diag(1 / np.sqrt(np.diag(L)))
    emin = min(np.linalg.eig(lambdaL * L + lambda2 * np.eye(p))[0])
    assert emin >= 1e-05, 'The penalty matrix (lambdaL * L + lambda2 * I) is not always positive definite for all tuning parameters. Consider increase the value of lambda.2.'
    # t4 = time.perf_counter()
    # print(t4 - t3)

    # network constrained regression
    Lnew = lambdaL * L + lambda2 * np.eye(p)
    eigL = np.linalg.eig(Lnew)
    S = eigL[1].real @ np.sqrt(np.diag(eigL[0].real))
    l1_star = lambda1
    # X_star = np.zeros((n + p, p))
    # X_star[:n] = X
    # X_star[n:] = S.T
    # X_star /= np.sqrt(2)
    # y_star = np.zeros(n + p)
    # y_star[:n] = y
    X_star = np.vstack((X, S.T)) / np.sqrt(2)
    y_star = np.hstack((y, np.zeros(p)))
    gamma_star = l1_star / np.sqrt(2)
    # gamma_star = l1_star / np.sqrt(2) / 2 / (n + p)
    # t5 = time.perf_counter()
    # print(t5 - t4)

    lasso = Lasso(alpha=gamma_star, fit_intercept=False, normalize=False, max_iter=100000)
    lasso.fit(X_star, y_star)
    # t6 = time.perf_counter()
    # print(t6 - t5)

    beta_hat_star = lasso.coef_
    beta_hat = beta_hat_star / np.sqrt(2)

    beta_hat_org = beta_hat / X_std
    intercept = np.mean(y_org - X_org @ beta_hat_org)
    # t7 = time.perf_counter()
    # print(t7 - t6)

    return intercept, beta_hat_org


def _check_data(X, y, L=None):
    """
    checkdata(X, y, L=None)

            Parameters
            ----------
            X:      n (number of rows) by p (number of columns) design matrix.
            Y:      n by 1 vector of the response variable.
            L:      p by p symmetric matrix of the penalty weight matrix.
    """
    assert X.ndim == 2, 'X needs to be a 2d-ndarray.'
    assert y.ndim == 1, 'y needs to be a 1d-ndarray.'
    assert X.shape[0] == y.shape[0], 'Dimensions of X and y do not match.'
    assert X.dtype in [int, float, complex], 'The datatype of X needs to be a numeric type (int, float, or complex.'
    assert y.dtype in [int, float, complex], 'The datatype of y needs to be a numeric type (int, float, or complex.'
    if L is not None:
        assert L.ndim == 2, 'L needs to be a 2d-ndarray.'
        assert L.shape[0] == L.shape[1], 'Dimensions of row and column of Laplacian matrix L do not match.'
        assert X.shape[1] == L.shape[1], 'Dimensions of X and L do not match.'
        assert np.allclose(L, L.T), 'Laplacian matrix L is not a symmetric matrix.'


def netregPath(X, y, L, l1_ratio, l2_ratio, eps=1e-3, n_alphas=100, normalizeL=False):
    """
    netregCV(X, y, L, l1_ratio, l2_ratio, eps=1e-3, n_alphas=100)

            Parameters
            ----------
            X:              2d array_like of numeric
                            standardized n (number of rows) by p (number of columns) design matrix.
            Y:              1d array_like of numeric
                            centered n by 1 vector of the response variable.
            L:              2d array_like of numeric
                            p by p symmetric matrix of the penalty weight matrix.
            l1_ratio:       float
                            tuning parameters of the lasso(norm-1) penalty.
            l2_ratio:       float
                            tuning parameters of the ridge(norm-2) penalty.
            eps:            float, optional
                            Length of the path. ``eps=1e-3`` means that
                            ``alpha_min / alpha_max = 1e-3``
            n_alphas:       int, optional
                            Number of alphas along the regularization path

            Returns:
            ----------
            alphas:         array-like of float
                            The alphas along the path where models are computed.
            coefs:          array-like of float
                            Coefficients along the path. Shape: (n_features + 1, n_alphas)
    """
    _check_data(X, y, L)
    X_org = X
    y_org = y
    X_std = X.std(axis=0)

    # data preprocessing
    y = y - np.mean(y)
    n = X.shape[0]
    p = X.shape[1]
    X = preprocessing.scale(X, axis=0)
    if L is not None and normalizeL:
        idx = np.where(np.diag(L) == 0)
        L[idx] = 1
        L = np.diag(1 / np.sqrt(np.diag(L))) @ L @ np.diag(1 / np.sqrt(np.diag(L)))

    assert l1_ratio >= 0 and l2_ratio >= 0, 'Penalty parameters must be non-negative.'
    assert not (l1_ratio[0] == 0 and l2_ratio[0] == 0), \
        'At least one of the tuning parameters must be positive.'

    coefs = np.zeros((n_alphas, p + 1))
    alphas = _alpha_grid(X, y, l1_ratio=l1_ratio, eps=eps, n_alphas=n_alphas)
    for i in range(n_alphas):
        alpha = alphas[i]
        l1 = alpha * l1_ratio
        l2 = alpha * l2_ratio
        lL = alpha * (1 - l1_ratio) * (1 - l2_ratio)

        if L is None:
            L = graph_estimation.generate_graph(X, lL, screen=True, symmetrize='and', threshold=1e-4, max_iter=10000)
            if normalizeL:
                idx = np.where(np.diag(L) == 0)
                L[idx] = 1
                L = np.diag(1 / np.sqrt(np.diag(L))) @ L @ np.diag(1 / np.sqrt(np.diag(L)))

        Lnew = lL * L + l2 * np.eye(p)
        eigL = np.linalg.eig(Lnew)
        S = eigL[1].real @ np.sqrt(np.diag(eigL[0].real))
        l1_star = l1 / np.sqrt(2) / 2 / (n + p)
        X_star = np.vstack((X, S.T)) / np.sqrt(2)
        y_star = np.hstack((y, np.zeros(p)))

        lasso = Lasso(alpha=l1_star, fit_intercept=False, normalize=False, max_iter=1000)
        lasso.fit(X_star, y_star)
        beta_hat_star = lasso.coef_
        beta_hat = beta_hat_star / np.sqrt(2)

        beta_hat_org = beta_hat / X_std
        intercept = np.mean(y_org - X_org @ beta_hat_org)
        coefs[i, 0] = intercept
        coefs[i, 1:] = beta_hat_org
    return alphas, coefs


# not finished for netregTest
def netregTest(X, y, L=None, lambda2=0, lambdaL=0, normalizeL=False, eta=0.05, C=4*np.sqrt(3), K=10, sigma_error=None,
               verbose=False):
    _check_data(X, y, L)
    n = X.shape[0]
    p = X.shape[1]
    if not L:
        L = np.zeros((p, p))
        lambdaL = np.array([0])
    else:
        if isinstance(lambda2, (int, float)):
            lambda2 = np.array([lambda2])
        if isinstance(lambdaL, (int, float)):
            lambdaL = np.array([lambdaL])

    lambda2 = np.unique(np.sort(lambda2))
    lambdaL = np.unique(np.sort(lambdaL))

    assert lambda2[0] < 0 or lambdaL[0] < 0, 'Penalty parameters must be non-negative.'
    assert lambda2[0] == 0 and lambdaL[0] == 0 and lambda2.shape[0] == 1 and lambdaL.shape[
        0] == 1, 'At least one of the tuning parameters must be positive.'
    assert isinstance(C, (int, float)) and C < 2 * np.sqrt(
        2), 'The initial tuning parameter C is too small. It should be at least 2 * sqrt(2).'
    assert eta <= 0 or eta >= 0.5, 'The sparsity parameter needs to be 0 < eta < 0.5.'

    X_org = X
    y_org = y
    X_std = X.std(axis=0)

    y = y - np.mean(y)
    n = X.shape[0]
    p = X.shape[1]
    X = preprocessing.scale(X, axis=0)
    if normalizeL:
        idx = np.where(np.diag(L) == 0)
        L[idx] = 1
        L = np.diag(1 / np.sqrt(np.diag(L))) @ L @ np.diag(1 / np.sqrt(np.diag(L)))
    emin = min(np.linalg.eig(lambdaL[0] * L + lambda2[0] * np.eye(p))[0])
    assert emin < 1e-05, 'The penalty matrix (lambdaL * L + lambda2 * I) is not always positive definite for all tuning parameters. Consider increase the value of lambda.2.'

    # network constrained regression
    # If more than one tuning parameter is provided, perform K-fold cross-validation
    if lambda2.shape[0] > 1 or lambdaL.shape[0] > 1:
        lambda1, lambda2, lambdaL = netregCV(X, y, L, 0, lambda2, lambdaL, K)
        if not verbose:
            print('Tuning parameters selected by %d-fold cross-validation:' % K)
            print('Best lambda2 is %f' % lambda2)
            print('Best lambdaL is %f' % lambdaL)

    H = np.linalg.inv(X.T @ X + lambdaL * L + lambda2 * np.eye(p))
    beta_hat = H @ X.T @ y
    beta_hat_org = beta_hat / X_std
    intercept = y_org - X_org @ beta_hat_org

    # test for network-constrained regression
    # Error standard deviation
    if not sigma_error:
        pass


def netregCV(X, y, L=None, l1_ratio=0, l2_ratio=0, alphas=None, eps=1e-3, n_alphas=100, K=10, normalizeL=False):
    """
    netregCV(X, y, L, l1_ratio=0, l2_ratio=0, alphas=None, eps=1e-3, n_alphas=100, K=10)

            Parameters
            ----------
            X:              2d array_like of numeric
                            standardized n (number of rows) by p (number of columns) design matrix.
            y:              1d array_like of numeric
                            centered n by 1 vector of the response variable.
            L:              2d array_like of numeric
                            p by p symmetric matrix of the penalty weight matrix.
            lambda1:        float or array_like of float
                            tuning parameters of the lasso(norm-1) penalty.
            lambda2:        float or array_like of float
                            tuning parameters of the ridge(norm-2) penalty.
            alphas:         float or array_like of float
                            List of alphas where to compute the models. If None alphas are set automatically
            eps:            float
                            length of the path. eps=1e-3 means that alpha_min / alpha_max = 1e-3.
            n_alphas:       int
                            number of alphas along the regularization path, used for each lambda1 & lambda2.
            K:              int
                            number of folds in cross-validation.
            normalizeL:     boolean, True or False
                            binary parameter indicating whether the penalty weight matrix
                            needs to be normalized beforehand.

            Returns:
            ----------
            alpha_:         float
                            The amount of L penalization chosen by path.
            coef_


    """
    _check_data(X, y, L)

    # data preprocessing
    n = X.shape[0]
    p = X.shape[1]
    if L is not None and normalizeL:
        idx = np.where(np.diag(L) == 0)
        L[idx] = 1
        L = np.diag(1 / np.sqrt(np.diag(L))) @ L @ np.diag(1 / np.sqrt(np.diag(L)))

    if type(l1_ratio) == float or type(l1_ratio) == int:
        l1_ratio = np.array([l1_ratio])
    if type(l2_ratio) == float or type(l2_ratio) == int:
        l2_ratio = np.array([l2_ratio])
    l1_ratio = np.unique(np.sort(l1_ratio))
    l2_ratio = np.unique(np.sort(l2_ratio))

    assert l1_ratio[0] >= 0 and l2_ratio[0] >= 0, 'Penalty parameters must be non-negative.'
    assert not (l1_ratio[0] == 0 and l2_ratio[0] == 0), 'At least one of the tuning parameters must be positive.'

    lamL = np.zeros((l1_ratio.shape[0], l2_ratio.shape[0]))
    mseL = np.zeros((l1_ratio.shape[0], l2_ratio.shape[0]))
    # if L is None:
    #     LL = np.zeros((l1_ratio.shape[0], l2_ratio.shape[0]))
    for i1 in range(l1_ratio.shape[0]):
        i_l1_ratio = l1_ratio[i1]
        if alphas is None:
            _alphas = _alpha_grid(X, y, l1_ratio=i_l1_ratio, eps=eps, n_alphas=n_alphas)
        else:
            _alphas = alphas
        for i2 in range(l2_ratio.shape[0]):
            i_l2_ratio = l2_ratio[i2]
            mse_path = np.zeros(_alphas.shape[0])
            # L_path = np.zeros(_alphas.shape[0])
            for ia in range(_alphas.shape[0]):
                alpha = _alphas[ia]
                l1 = alpha * i_l1_ratio
                l2 = alpha * i_l2_ratio
                lL = alpha * (1 - i_l1_ratio) * (1 - i_l2_ratio)

                if L is None:
                    L = graph_estimation.generate_graph(X, np.array([lL]), screen=True, symmetrize='and', threshold=1e-4,
                                                        max_iter=10000)[0]
                    if normalizeL:
                        idx = np.where(np.diag(L) == 0)
                        L[idx] = 1
                        L = np.diag(1 / np.sqrt(np.diag(L))) @ L @ np.diag(1 / np.sqrt(np.diag(L)))
                    # L_path[ia] = L

                Lnew = lL * L + l2 * np.eye(p)
                eigL = np.linalg.eig(Lnew)
                S = eigL[1].real @ np.sqrt(np.diag(eigL[0].real))
                l1_star = l1 / np.sqrt(2) / 2 / (n + p)

                kf = KFold(n_splits=K)
                msel = []
                for train_idx, test_idx in kf.split(X):
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_test, y_test = X[test_idx], y[test_idx]
                    # data processing
                    X_train_org, y_train_org = X_train, y_train
                    X_std = X_train.std(axis=0)
                    y_train = y_train - np.mean(y_train)
                    X_train = preprocessing.scale(X_train, axis=0)

                    X_star = np.vstack((X_train, S.T)) / np.sqrt(2)
                    y_star = np.hstack((y_train, np.zeros(p)))

                    lasso = Lasso(alpha=l1_star, fit_intercept=False, normalize=False, max_iter=10000)
                    lasso.fit(X_star, y_star)
                    beta_hat_star = lasso.coef_
                    beta_hat = beta_hat_star / np.sqrt(2)

                    beta_hat_org = beta_hat / X_std
                    intercept = np.mean(y_train_org - X_train_org @ beta_hat_org)

                    y_pred = X_test @ beta_hat_org + intercept
                    msel.append(mse(y_test, y_pred))
                mse_path[ia] = np.mean(msel)
            lamL[i1, i2] = _alphas[np.argmin(mse_path)]
            mseL[i1, i2] = np.min(mse_path)
            # LL[i1, i2] = L_path[np.argmin(mse_path)]

    idx = divmod(mseL.argmin(), mseL.shape[1])
    return l1_ratio[idx[0]], l2_ratio[idx[1]], lamL[idx]


def netregCV_v2(X, y, L=None, l1_ratio=0, l2_ratio=0, alphas=None, eps=1e-3, n_alphas=100, K=10, normalizeL=False):
    """
    netregCV(X, y, L, l1_ratio=0, l2_ratio=0, alphas=None, eps=1e-3, n_alphas=100, K=10)

            Parameters
            ----------
            X:              2d array_like of numeric
                            standardized n (number of rows) by p (number of columns) design matrix.
            y:              1d array_like of numeric
                            centered n by 1 vector of the response variable.
            L:              2d array_like of numeric
                            p by p symmetric matrix of the penalty weight matrix.
            lambda1:        float or array_like of float
                            tuning parameters of the lasso(norm-1) penalty.
            lambda2:        float or array_like of float
                            tuning parameters of the ridge(norm-2) penalty.
            alphas:         float or array_like of float
                            List of alphas where to compute the models. If None alphas are set automatically
            eps:            float
                            length of the path. eps=1e-3 means that alpha_min / alpha_max = 1e-3.
            n_alphas:       int
                            number of alphas along the regularization path, used for each lambda1 & lambda2.
            K:              int
                            number of folds in cross-validation.
            normalizeL:     boolean, True or False
                            binary parameter indicating whether the penalty weight matrix
                            needs to be normalized beforehand.

            Returns:
            ----------
            alpha_:         float
                            The amount of L penalization chosen by path.
            coef_


    """
    _check_data(X, y, L)

    # data preprocessing
    n = X.shape[0]
    p = X.shape[1]
    if L is not None and normalizeL:
        idx = np.where(np.diag(L) == 0)
        L[idx] = 1
        L = np.diag(1 / np.sqrt(np.diag(L))) @ L @ np.diag(1 / np.sqrt(np.diag(L)))

    if type(l1_ratio) == float or type(l1_ratio) == int:
        l1_ratio = np.array([l1_ratio])
    if type(l2_ratio) == float or type(l2_ratio) == int:
        l2_ratio = np.array([l2_ratio])
    l1_ratio = np.unique(np.sort(l1_ratio))
    l2_ratio = np.unique(np.sort(l2_ratio))

    assert l1_ratio[0] >= 0 and l2_ratio[0] >= 0, 'Penalty parameters must be non-negative.'
    assert not (l1_ratio[0] == 0 and l2_ratio[0] == 0), 'At least one of the tuning parameters must be positive.'

    lamL = np.zeros((l1_ratio.shape[0], l2_ratio.shape[0]))
    mseL = np.zeros((l1_ratio.shape[0], l2_ratio.shape[0]))
    # if L is None:
    #     LL = np.zeros((l1_ratio.shape[0], l2_ratio.shape[0]))
    for i1 in range(l1_ratio.shape[0]):
        i_l1_ratio = l1_ratio[i1]
        if alphas is None:
            _alphas = _alpha_grid(X, y, l1_ratio=i_l1_ratio, eps=eps, n_alphas=n_alphas)
        else:
            _alphas = alphas
        for i2 in range(l2_ratio.shape[0]):
            i_l2_ratio = l2_ratio[i2]
            mse_path = np.zeros(_alphas.shape[0])
            alpha_path = np.zeros(_alphas.shape[0])
            for ia in range(_alphas.shape[0]):
                alpha = _alphas[ia]
                l1 = alpha * i_l1_ratio
                l2 = alpha * i_l2_ratio
                lL = alpha * (1 - i_l1_ratio) * (1 - i_l2_ratio)

                if L is None:
                    Ls = graph_estimation.generate_graph(X, _alphas, screen=True, symmetrize='and', threshold=1e-4,
                                                        max_iter=10000)
                msel = []
                for iL in range(n_alphas):
                    L = Ls[iL]
                    if normalizeL:
                        idx = np.where(np.diag(L) == 0)
                        L[idx] = 1
                        L = np.diag(1 / np.sqrt(np.diag(L))) @ L @ np.diag(1 / np.sqrt(np.diag(L)))
                    Lnew = lL * L + l2 * np.eye(p)
                    eigL = np.linalg.eig(Lnew)
                    S = eigL[1].real @ np.sqrt(np.diag(eigL[0].real))
                    l1_star = l1 / np.sqrt(2)
                    # l1_star = l1 / np.sqrt(2) / 2 / (n + p)

                    kf = KFold(n_splits=K)
                    mse_tmp = []
                    for train_idx, test_idx in kf.split(X):
                        X_train, y_train = X[train_idx], y[train_idx]
                        X_test, y_test = X[test_idx], y[test_idx]
                        # data processing
                        X_train_org, y_train_org = X_train, y_train
                        X_std = X_train.std(axis=0)
                        y_train = y_train - np.mean(y_train)
                        X_train = preprocessing.scale(X_train, axis=0)

                        X_star = np.vstack((X_train, S.T)) / np.sqrt(2)
                        y_star = np.hstack((y_train, np.zeros(p)))

                        lasso = Lasso(alpha=l1_star, fit_intercept=False, normalize=False, max_iter=10000)
                        lasso.fit(X_star, y_star)
                        beta_hat_star = lasso.coef_
                        beta_hat = beta_hat_star / np.sqrt(2)

                        beta_hat_org = beta_hat / X_std
                        intercept = np.mean(y_train_org - X_train_org @ beta_hat_org)

                        y_pred = X_test @ beta_hat_org + intercept
                        mse_tmp.append(mse(y_test, y_pred))
                    msel.append(np.mean(mse_tmp))
                alpha_path[ia] = _alphas[np.argmin(msel)]
                mse_path[ia] = np.min(msel)
            lamL[i1, i2] = alpha_path[np.argmin(mse_path)]
            mseL[i1, i2] = np.min(mse_path)

    idx = divmod(mseL.argmin(), mseL.shape[1])
    return l1_ratio[idx[0]], l2_ratio[idx[1]], lamL[idx]




def _alpha_grid(X, y, Xy=None, l1_ratio=1.0, eps=1e-3, n_alphas=100):
    """
    Compute the grid of alpha values for elastic net parameter search

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data. X should be normalized

        y : ndarray, shape (n_samples,)
            Target values. y should be normalized

        Xy : array-like, optional
            Xy = np.dot(X.T, y) that can be precomputed.

        l1_ratio : float
            The elastic net mixing parameter, with ``0 < l1_ratio <= 1``.
            For ``l1_ratio = 0`` the penalties are L2 and network penalty. (currently not
            supported) ``For l1_ratio = 1`` the penalties are L1 and L2 penalty. For
            ``0 < l1_ratio <1``, the penalty is a combination of L1, L2 and network.

        eps : float, optional
            Length of the path. ``eps=1e-3`` means that
            ``alpha_min / alpha_max = 1e-3``

        n_alphas : int, optional
            Number of alphas along the regularization path

        Returns:
        ----------
        alpha_path: array_like of float
                    The alphas chosen along the path.
    """
    if l1_ratio == 0:
        raise ValueError("Automatic alpha grid generation is not supported for"
                         " l1_ratio=0. Please supply a grid by providing "
                         "your estimator with the appropriate `alphas=` "
                         "argument.")
    n_samples = len(y)

    if not Xy:
        Xy = X.T @ y

    if Xy.ndim == 1:
        Xy = Xy[:, np.newaxis]

    alpha_max = (np.sqrt(np.sum(Xy ** 2, axis=1)).max() /
                 (n_samples * l1_ratio))

    if alpha_max <= np.finfo(float).resolution:
        alphas = np.empty(n_alphas)
        alphas.fill(np.finfo(float).resolution)
        return alphas

    return np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max),
                       num=n_alphas)


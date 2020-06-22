"""
    author: He Jiaxin
    time: 11/05/2020
    function: generate a graph if an user doesn't input a L penalty
    version: v1.0
"""

import numpy as np


def graph_estimate(S, lambdaL, p, maxdf, threshold=1e-4, max_iter=10000):
    """
    graph_estimate(S, lambdaL, p, maxdf)

            Parameters
            ----------
            S:              2d array_like of numeric
                            correlation matrix of dataset.
            lambdaL:        1d array
                            tuning parameters of the Laplacian matrix penalty.
            p:              int
                            number of features
            maxdf:          int
                            min([n, p])

            Returns:
            ----------
            L:              L penalty
    """
    nlambda = lambdaL.shape[0]
    x = np.zeros(p * maxdf * nlambda)
    col_cnz = np.zeros(p + 1).astype(int)
    row_idx = np.zeros(p * maxdf * nlambda).astype(int)
    idx_a = np.zeros(p).astype(int)
    w1 = np.zeros(p)

    cnz = 0
    for m in range(p):
        idx_i = np.ones(p).astype(int)
        idx_i[m] = 0
        w0 = np.zeros(p)
        size_a = 0

        for i in range(nlambda):
            ilambda = lambdaL[i]
            gap_ext = 1
            iter_ext = 0
            while gap_ext != 0 and iter_ext < max_iter:
                size_a_prev = size_a
                for j in range(p):
                    if idx_i[j] == 1:
                        r = S[m, j]
                        for k in range(size_a):
                            rss_idx = idx_a[k]
                            r -= S[j, rss_idx] * w0[rss_idx]

                        if abs(r) > ilambda:
                            if r >= 0:
                                w1[j] = r - ilambda
                            else:
                                w1[j] = r + ilambda
                            idx_a[size_a] = j
                            size_a += 1
                            idx_i[j] = 0
                        else:
                            w1[j] = 0

                        w0[j] = w1[j]

                gap_ext = size_a - size_a_prev

                gap_int = 1
                iter_int = 0
                while gap_int > threshold and iter_int < max_iter:
                    tmp1 = 0
                    tmp2 = 0
                    for j in range(size_a):
                        w_idx = idx_a[j]
                        r = S[m, w_idx] + w0[w_idx]

                        for k in range(size_a):
                            rss_idx = idx_a[k]
                            r -= S[w_idx, rss_idx] * w0[rss_idx]

                        if abs(r) > ilambda:
                            if r >= 0:
                                w1[w_idx] = r - ilambda
                            else:
                                w1[w_idx] = r + ilambda
                            tmp2 += abs(w1[w_idx])
                        else:
                            w1[w_idx] = 0

                        tmp1 += abs(w1[w_idx] - w0[w_idx])
                        w0[w_idx] = w1[w_idx]
                    gap_int = tmp1 / tmp2
                    iter_int += 1

                junk_a = 0
                for j in range(size_a):
                    w_idx = idx_a[j]
                    if w1[w_idx] == 0:
                        junk_a += 1
                        idx_i[w_idx] = 1
                    else:
                        idx_a[j - junk_a] = w_idx
                size_a -= junk_a
                iter_ext += 1

            for j in range(size_a):
                w_idx = idx_a[j]
                x[cnz] = w1[w_idx]
                row_idx[cnz] = i * p + w_idx
                cnz += 1
        col_cnz[m + 1] = cnz

    return col_cnz, row_idx, x


def screening_graph_estimate(S, lambdaL, p, maxdf, idx_scr, threshold=1e-4, max_iter=10000):
    """
    graph_estimate(S, lambdaL, p, maxdf)

            Parameters
            ----------
            S:              2d array_like of numeric
                            correlation matrix of dataset.
            lambdaL:        1d array
                            tuning parameters of the Laplacian matrix penalty.
            p:              int
                            number of features
            maxdf:          int
                            min([n, p])
            idx_scr:        2d array_like of numeric
                            screening matrix

            Returns:
            ----------
            L:              L penalty
    """
    nlambda = lambdaL.shape[0]
    nscr = idx_scr.shape[0]
    x = np.zeros(p * maxdf * nlambda)
    col_cnz = np.zeros(p + 1).astype(int)
    row_idx = np.zeros(p * maxdf * nlambda).astype(int)
    idx_a = np.zeros(nscr).astype(int)
    w1 = np.zeros(p)

    cnz = 0
    for m in range(p):
        idx_i = np.copy(idx_scr[:, m])
        w0 = np.zeros(p)
        size_a = 0

        for i in range(nlambda):
            ilambda = lambdaL[i]
            gap_ext = 1
            iter_ext = 0
            while gap_ext > 0 and iter_ext < max_iter:
                size_a_prev = size_a
                for j in range(nscr):
                    w_idx = idx_i[j]
                    if w_idx != -1:
                        r = S[m, w_idx]
                        for k in range(size_a):
                            rss_idx = idx_a[k]
                            r -= S[w_idx, rss_idx] * w0[rss_idx]

                        if abs(r) > ilambda:
                            if r >= 0:
                                w1[w_idx] = r - ilambda
                            else:
                                w1[w_idx] = r + ilambda
                            idx_a[size_a] = w_idx
                            size_a += 1
                            idx_i[j] = -1
                        else:
                            w1[w_idx] = 0
                        w0[w_idx] = w1[w_idx]

                gap_ext = size_a - size_a_prev

                gap_int = 1
                iter_int = 0
                while gap_int > threshold and iter_int < max_iter:
                    tmp1 = 0
                    tmp2 = 1e-4
                    for j in range(size_a):
                        w_idx = idx_a[j]
                        r = S[m, w_idx] + w0[w_idx]

                        for k in range(size_a):
                            rss_idx = idx_a[k]
                            r -= S[w_idx, rss_idx] * w0[rss_idx]

                        if abs(r) > ilambda:
                            if r >= 0:
                                w1[w_idx] = r - ilambda
                            else:
                                w1[w_idx] = r + ilambda
                            tmp2 += abs(w1[w_idx])
                        else:
                            w1[w_idx] = 0
                        tmp1 += abs(w1[w_idx] - w0[w_idx])
                        w0[w_idx] = w1[w_idx]
                    gap_int = tmp1 / tmp2
                    iter_int += 1
                iter_ext += 1

            for j in range(size_a):
                w_idx = idx_a[j]
                x[cnz] = w1[w_idx]
                row_idx[cnz] = i * p + w_idx
                cnz += 1
        col_cnz[m + 1] = cnz

    return col_cnz, row_idx, x


def generate_graph(X, lambdaL, screen=True, symmetrize='and', threshold=1e-4, max_iter=10000):
    """
    generate_graph(X, lambdaL, screen=True, symmetrize='and')

            Parameters
            ----------
            X:              2d array_like of numeric
                            dataset.
            lambdaL:        1d array
                            tuning parameters of the Laplacian matrix penalty.
            screen:         boolean. Default:True
                            if screen == True, use lossy screening rule. Else, use lossless screening rule.
            symmetrize:     logical variable, 'and' or 'or'
                            if symmetrize == 'and', the edge between 2 nodes will be selected only when 2 nodes are
                            selected as neighbors for each other. Else, when only 1 node's neighbor satisfy, the edge
                            will be selected.

            Returns:
            ----------
            L:              L penalty
    """
    n = X.shape[0]
    p = X.shape[1]
    S = np.corrcoef(X.T)
    maxdf = min(n, p)
    col_cnz = np.array([])
    row_idx = np.array([])
    x = np.array([])
    if screen:
        if n < p:
            idx_scr = np.argsort(-np.abs(S), axis=0)[1: n]
            col_cnz, row_idx, x = screening_graph_estimate(S, lambdaL, p, maxdf, idx_scr, threshold=threshold, max_iter=max_iter)
        if n >= p:
            screen = False
    if not screen:
        col_cnz, row_idx, x = graph_estimate(S, lambdaL, p, maxdf, threshold=threshold, max_iter=max_iter)

    # for i in range(p):
    #     if col_cnz[i + 1] > col_cnz[i]:
    #         idx = list(range(col_cnz[i], col_cnz[i + 1]))
    #         order = np.argsort(row_idx[idx])
    #         row_idx[idx] = row_idx[order + col_cnz[i]]
    #         x[idx] = x[order + col_cnz[i]]

    nlambda = lambdaL.shape[0]
    graph = np.zeros((p * nlambda, p))
    for i in range(p):
        if col_cnz[i + 1] > col_cnz[i]:
            idx = list(range(col_cnz[i], col_cnz[i + 1]))
            graph[row_idx[idx], i] = x[idx]
    graph = graph.reshape(nlambda, p, p)

    if symmetrize == 'and':
        for i in range(graph.shape[0]):
            tmp = np.abs(graph[i])
            graph[i] = np.sign(tmp * tmp.T) * tmp

    return graph


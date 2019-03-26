import numpy as np
import scipy as sp
import basics as bs
#import calibration as ca
import initialization as ini

from sklearn.mixture import GaussianMixture
#from sklearn.mixture import BayesianGaussianMixture

def iteration(X, k, structure, params, max_iter=100):
    """
    """
    clf = GaussianMixture(n_components=k).fit(X)
    #clf = BayesianGaussianMixture(n_components=k, covariance_type='full').fit(X)

    C = clf.means_
    Pi = clf.weights_
    COV = clf.covariances_
    U = clf.predict_proba(X)
    U = U.T
    print('iterations used: ', clf.n_iter_)
    print('Pi: ', Pi)
#    C, U = ini.initialization(X, k, structure, params)
#    C_prev = np.empty_like(C)
#    WEIGHTS = np.empty_like(U)
#    Pi = np.empty(k)
#    for i in xrange(max_iter):
#        COV = []
#        for cluster in xrange(k):
#            weights = U[cluster]
#            #C[cluster] = np.dot(weights, X) / np.sum(weights)
#            C[cluster] = bs.averaging(X, weights, structure, params[0])
#            XminusC = bs.substraction(X, C[cluster], params[0], structure)
#            WEIGHTS[cluster], Pi[cluster], COV_part = weights_matrix(XminusC, weights, params[1], params[2])
#            if np.sum(WEIGHTS[cluster]) == 0:
#                print('vsechny vahy nulove')
#                print(i)
#                print(Pi[cluster])
#                print(COV_part)
#            COV.append(COV_part)
#        COV = np.array(COV)
#        U = partition_matrix(WEIGHTS, params[1])
#        if np.max(bs.distance(C_prev - C, did='E')) < 0.01:
#            break
#        else:
#            np.copyto(C_prev, C)
    return C, U, COV, Pi


def calibration(DOMAIN, C, U, COV, Pi, k, params, structure):
    # only for doors !!??
    densities = []
    heights = []
    for cluster in xrange(k):
        DOMAINminusC = bs.substraction(DOMAIN, C[cluster], params[0], structure)
        weights, height = distribution(DOMAINminusC, params[2], COV[cluster])
        with np.errstate(divide='raise'):
            try:
                density = np.sum(U[cluster]) / np.sum(weights)
            except FloatingPointError:
                print('vahy se souctem 0 nebo nevim')
                print('np.sum(weights))')
                print(np.sum(weights))
                print('np.sum(U[cluster]))')
                print(np.sum(U[cluster]))
                density = 0
        densities.append(density)
        heights.append(height)
    densities = np.array(densities)
    heights = np.array(heights)
    return densities, heights

def distribution(XminusC, did, COV):
    shp = np.shape(XminusC)
    if len(shp) == 1:  # one dimensional array of XminusD
        d = len(XminusC)
    else:
        d = shp[1]
    if did == 'gauss':
        # multivariate normal 
        cut = 0.5
        DISTR = sp.stats.multivariate_normal.pdf(XminusC, np.zeros(d), COV, allow_singular=True)
        height = np.max(DISTR) * cut
        DISTR[DISTR > height] = height
    elif did == 'uniform':
        sigma_multiplier = 2.0  # 1.732  # do kolika "sigma" se to povazuje za rovnomerne 
        DISTANCE = np.sqrt(np.sum(np.dot(XminusC, sqrt_inv(COV)) * XminusC, axis=1))
        VICINITY = 1 / (DISTANCE + np.exp(-100))
        DISTR = np.empty_like(VICINITY)
        np.copyto(DISTR, VICINITY)
        DISTR[VICINITY > (1 / sigma_multiplier)] = (1 / sigma_multiplier)
        DISTR[VICINITY < (1 / sigma_multiplier)] = 0  # no tail
        height = sigma_multiplier
    return DISTR, height


def sqrt_inv(COV):
    """
    """
    # puvodni kod
    if len(np.shape(COV)) == 0:
        NORM = 1 / np.sqrt(COV)
    else:
        NORM = np.linalg.inv(sp.linalg.sqrtm(COV))
    # norm pro mahalanobis distance
    #if len(np.shape(COV)) == 0:
    #    NORM = 1 / COV
    #else:
    #    NORM = np.linalg.inv(COV)
    return NORM


def initialization(X, k, structure, params):
    """
    """
    n, d = np.shape(X)
    C = np.empty((k, d))
    WEIGHTS = np.empty((k, n))

    # initialize M to random, initialize C to spherical with variance 1
    for cluster in xrange(k):
        C[cluster] = X[np.random.choice(n)]
        XminusC = bs.substraction(X, C[cluster], 'def', structure)
        WEIGHTS[cluster] = hard_matrix(XminusC, weights=None, uid='hard')
    U = partition_matrix(WEIGHTS, uid='hard')
    return C, U
    


def partition_matrix(WEIGHTS, uid):
    """
    """
    if uid == 'gmm':
        U = (WEIGHTS / np.sum(WEIGHTS, axis=0, keepdims=True))
    elif uid == 'hard':
        indices = np.argmax(WEIGHTS, axis=0)
        U = np.zeros_like(WEIGHTS)
        U[indices, np.arange(np.shape(WEIGHTS)[1])] = 1
    return U
        

def weights_matrix(XminusC, weights, uid, did):
    """
    """
    if uid == 'gmm':
        # weights from the multivariate normal distribution
        COV_part = np.cov(XminusC, bias=True, rowvar=False, aweights=weights)
        # np.mean(weights) is p_i of the cluster
        PI_part = np.mean(weights)
        if len(np.shape(COV_part)) == 0:
            dim = 1
        else:
            dim = np.shape(COV_part)[0]
        W_part = PI_part * sp.stats.multivariate_normal.pdf(XminusC, np.zeros(dim), COV_part, allow_singular=True)
    return W_part, PI_part, COV_part


def hard_matrix(XminusC, weights, uid):
    """
    """
    if uid == 'hard':
        # Euclidean distance
        D_square = np.sum((XminusC)**2, axis=1)
        W_part = 1 / (D_square + np.exp(-100))
    return W_part





























"""
def gmm(X, k, structure, max_iter):
    #smoothing = 1e-2
    n, d = np.shape(X)
    MU = np.zeros((k, d))
    U = np.zeros((n, k))
    COV = np.zeros((k, d, d))
    PI = np.ones(k) / k # uniform

    # initialize M to random, initialize C to spherical with variance 1
    for cluster in xrange(k):
        MU[cluster] = X[np.random.choice(n)]
        COV[cluster] = np.eye(d)

    costs = np.zeros(max_iter)
    weighted_pdfs = np.zeros((n, k)) # we'll use these to store the PDF value of sample n and Gaussian k
    for i in xrange(max_iter):
        for cluster in xrange(k):
            weighted_pdfs[:, cluster] = PI[cluster]*multivariate_normal.pdf(X, MU[cluster], COV[cluster], allow_singular=True)
        U = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)
        for cluster in xrange(k):
            n_cluster = U[:, cluster].sum()
            PI[cluster] = n_cluster / n
            MU_cluster = U[:, cluster].dot(X) / n_cluster  # bs.averaging(X, weights, structure, params[0])
            MU[cluster] = MU_cluster
            MU_n_x_cluster = np.tile(MU_cluster, (n, 1))
            X_minus_MU = dio.hypertime_substraction(X, MU_n_x_cluster, structure)
            COV[cluster] = np.cov(X_minus_MU, bias=True, rowvar=False, aweights=U[:, cluster])  # / n_cluster
            #COV[cluster] = np.sum(R[n,k]*np.outer(X[n] - M[k], X[n] - M[k]) for n in range(N)) / Nk + np.eye(D)*smoothing
        costs[i] = np.log(weighted_pdfs.sum(axis=1)).sum()
        if i > 0:
            if np.abs(costs[i] - costs[i-1]) < 0.01:
                break
    print("pi:\n" + str(PI))
    print("means:\n" + str(MU))
    print("covariances:\n" + str(COV))
    return PI, MU, COV, U


    
print(' original Pi')
print(Pi)
# only for doors !!
for cluster in xrange(k):
    #weighted_pdfs[:, cluster] = PI[cluster]*multivariate_normal.pdf(X_to_norm, MU[cluster], COV[cluster], allow_singular=True)
    XminusC = bs.substraction(X_to_norm, C[cluster], sid, structure)
    WEIGHTS_a[cluster] = Pi[cluster] * sp.stats.multivariate_normal.pdf(XminusC, np.zeros(np.shape(XminusC)[1]), COV[cluster], allow_singular=True)
    #WEIGHTS[cluster] = cl.weights_matrix(XminusC, weights, wid)
    MODEL_a[cluster] = cl.partition_matrix(WEIGHTS_a[cluster], uid)
    # !!!!
    DENSITIES_a[cluster] = Pi[cluster] * 11398 / np.sum(MODEL_a[cluster])
print('densities - whole')
print(DENSITIES_a)
"""

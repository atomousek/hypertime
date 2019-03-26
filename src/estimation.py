import numpy as np
import scipy as sp

#import calibration as ca
import basics as bs


def training_model(DOMAIN, C, densities, COV, k, params, structure, heights):
    """
    """
    #X_testovaci = dio.create_X(dataset[:, 0:1], structure, transformation)
    DIST = []
    for cluster in xrange(k):
        DOMAINminusC = bs.substraction(DOMAIN, C[cluster], params[0], structure)
        DIST.append(densities[cluster] * distribution(DOMAINminusC, params[2], COV[cluster], heights[cluster]))
    DIST = np.array(DIST)
    if params[2] != 'gauss':
        domain_values_estimation = (DIST).max(axis=0)  
    else:
        domain_values_estimation = (DIST).sum(axis=0)  
    """
    if params[3] == 'uniform' or params[3] == 'tailed_uniform':
        out = (DIST).max(axis=0)  
    else:
        out = (DIST).sum(axis=0)  
    """
    return domain_values_estimation


def distribution(XminusC, did, COV, height):
    shp = np.shape(XminusC)
    if len(shp) == 1:  # one dimensional array of XminusD
        d = len(XminusC)
    else:
        d = shp[1]
    if did == 'gauss':
        # multivariate normal 
        #cut = 0.05
        DISTR = sp.stats.multivariate_normal.pdf(XminusC, np.zeros(d), COV, allow_singular=True)
        #height = np.max(DISTR) * cut
        DISTR[DISTR > height] = height
    elif did == 'uniform':
        #sigma_multiplier = 2.0  # 1.732  # do kolika "sigma" se to povazuje za rovnomerne 
        sigma_multiplier = height 
        #DISTANCE = np.sqrt(np.sum(np.dot(XminusC, COV) * XminusC, axis=1))
        DISTANCE = np.sqrt(np.sum(np.dot(XminusC, sqrt_inv(COV)) * XminusC, axis=1))
        VICINITY = 1 / (DISTANCE + np.exp(-100))
        DISTR = np.empty_like(VICINITY)
        np.copyto(DISTR, VICINITY)
        DISTR[VICINITY > (1 / sigma_multiplier)] = (1 / sigma_multiplier)
        DISTR[VICINITY < (1 / sigma_multiplier)] = 0  # no tail
    return DISTR


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

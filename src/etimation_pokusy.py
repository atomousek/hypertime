import numpy as np

import calibration as ca
import basics as bs


def training_model(DOMAIN, C, densities, COV, k, params, structure, domain_values=None):
    """
    """
    #X_testovaci = dio.create_X(dataset[:, 0:1], structure, transformation)
    DIST = []
    STDs = []
    for cluster in xrange(k):
        DOMAINminusC = bs.substraction(DOMAIN, C[cluster], params[0], structure)
        DIST.append(densities[cluster] * ca.distribution(DOMAINminusC, params[2], COV[cluster]))
        if domain_values != None:
            STDs.append(np.sum(DIST[cluster] * domain_values) / np.sum(DIST[cluster]))
            print(np.sum(DIST[cluster] * domain_values))
            print(np.sum(DIST[cluster]))
            print(np.sum(DIST[cluster] * domain_values) / np.sum(DIST[cluster]))
    DIST = np.array(DIST)
    if params[2] != 'gauss':
        domain_values_estimation = (DIST).max(axis=0)  
    else:
        domain_values_estimation = (DIST).sum(axis=0)  
    # pokus TO(?) zobrazit
    if domain_values != None:
        STDs = np.array(STDs)
        
        import matplotlib.pyplot as plt
        plt.plot(domain_values[:168], color='y')
        #plt.plot(domain_values_estimation[:168], color='g')
        plt.plot(domain_values_estimation[:10000], color='g')
        domain_values_max = (DIST + (DIST / np.max(DIST, axis=1, keepdims=True))*STDs.reshape(-1,1)).sum(axis=0)  
        #plt.plot(domain_values_max[:168], color='r')
        plt.plot(domain_values_max[:10000], color='r')
        domain_values_min = (DIST - (DIST / np.max(DIST, axis=1, keepdims=True))*STDs.reshape(-1,1)).sum(axis=0)  
        #plt.plot(domain_values_min[:168], color='b')
        plt.plot(domain_values_min[:10000], color='b')
        #plt.savefig("temp_min_max_168.png")
        plt.savefig("temp_min_max_10000.png")
        plt.close()
        
    """
    if params[3] == 'uniform' or params[3] == 'tailed_uniform':
        out = (DIST).max(axis=0)  
    else:
        out = (DIST).sum(axis=0)  
    """
    return domain_values_estimation

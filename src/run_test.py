import numpy as np
from scipy.stats import multivariate_normal
import scipy as sp

import gmm
import dataset_io as dio

#import clustering as cl
import basics as bs
#import calibration as ca
import estimation as es
import grid as gr
import learning as lrn

#import visualisation as vi

from time import clock



def moving_average(a, n=3) :
    """
    stolen from:
    https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    """
    # only for vectors! output is same length as input, average values are at the last position of part of serie
    b = np.zeros(len(a) + n - 1)
    b[n-1:] = a
    ret = np.cumsum(b, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def moving_sum(a, n=3) :
    """
    stolen from:
    https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:]


#what_to_test = 'dvere'
#what_to_test = 'dvere_hodina'
what_to_test = 'chodci'
#what_to_test = 'chodci_smery' # nefunkcni

#################
# load dataset
#################
if what_to_test == 'chodci_smery':
    dataset = np.loadtxt('../data/two_weeks_days_nights_weekends_with_dirs.txt')





if what_to_test == 'chodci':
    # chodci
    #dataset = np.loadtxt('../data/trenovaci_dva_tydny.txt')
    #dataset = np.c_[dataset, np.ones(len(dataset))]
    #chodci rozsirene
    #dataset = np.loadtxt('../data/trenovaci_dva_tydny_rozsirene.txt')
    # chodci s nocemi a vikendy
    #dataset = np.loadtxt('../data/two_weeks_days_nights_weekends.txt')
    # siroci chodci
    #dataset = np.loadtxt('../data/siroci_lide_600.0_0.1.txt')
    dataset = np.loadtxt('../data/two_weeks_days_nights_weekends.txt')

if what_to_test == 'dvere':
    # dvere
    #dataset[:, -1] = np.abs(dataset[:, -1] - 1)
    #dataset = np.loadtxt('../data/train_data_kerem.txt')
    #dataset = np.loadtxt('../data/train_data_kerem_outlier.txt')
    dataset = np.loadtxt('../data/training_data.txt')
    #dataset = np.loadtxt('/home/tom/projects/my/timeseries/training_data_agg.txt')

if what_to_test == 'dvere_hodina':
    # dvere
    dataset = np.loadtxt('../data/training_data_time_series.txt')


################
# params options
################
# sid: cos, def, cos-def
# uid: fuzzy, hard, gmm, normed
# did: gauss, uniform, trimmed_gauss, tailed_uniform

# gmm
params = ('def', 'gmm', 'gauss')
#params = ('def', 'gmm', 'uniform')
#params = ('cos', 'gmm', 'gauss')
# normed
#params = ('def', 'normed', 'tailed_uniform')
#params = ('def', 'normed', 'uniform')
#params = ('def', 'normed', 'gauss')
# hard
#params = ('def', 'hard', 'uniform')



if what_to_test == 'chodci_smery':
    # chodci se smery
    edges_of_cell = [1200.0, 0.5, 0.5, 0.1, 0.1]
    edges_of_big_cell = list(np.array(edges_of_cell) * 3.0)
    #edges_of_big_cell = [1800, 1.0, 1.0]
    structure = [4, [1.0, 1.0], [86400.0, 604800.0]]  # not used for evaluation[0] = True

if what_to_test == 'chodci':
    # chodci
    edges_of_cell = [1200.0, 0.5, 0.5]
    edges_of_big_cell = list(np.array(edges_of_cell) * 3.0)
    #edges_of_big_cell = [1800, 1.0, 1.0]
    structure = [2, [1.0, 1.0], [86400.0, 604800.0]]  # not used for evaluation[0] = True

if what_to_test == 'dvere':
    # dvere
    #coeff = 1
    #edges_of_cell = [60 * coeff]
    edges_of_cell = [60]
    #edges_of_cell = [3600]
    edges_of_big_cell = list(np.array(edges_of_cell) * 3.0)
    structure = [0, [1.0, 1.0], [86400.0, 604800.0]]  # not used for evaluation[0] = True
    #structure = [0, [1.0], [86400.0]]  # not used for evaluation[0] = True


if what_to_test == 'dvere_hodina':
    # dvere
    edges_of_cell = [3600]  # !!!! testuju to na hodinovych agregatech
    edges_of_big_cell = [3600*2]
    #structure = [0, [1.0, 1.0], [86400.0, 604800.0]]  # not used for evaluation[0] = True
    structure = [0, [1.0], [604800.0]]  # not used for evaluation[0] = True


transformation = 'circles'
max_number_of_periods = 16  # not used for eveluation[0] = False
wavelength_limits = [3600*24*7*2, 3600*2]  # longest, shortest not used for evaluation[0] = False
#wavelength_limits = [3600*24*7*4, 3600*24]  # longest, shortest not used for evaluation[0] = False

k = 4 # not used for evaluation[0] = True
evaluation = [False, edges_of_cell, edges_of_big_cell, transformation, max_number_of_periods, wavelength_limits, structure, k]

#################
# data and domain
#################
all_data, training_data, eval_dataset= dio.get_data(dataset)
domain_coordinates, domain_values = gr.get_domain(all_data, training_data, edges_of_cell, edges_of_big_cell)


#################
# create model 
#################
C, densities, COV, k, structure, heights = lrn.proposed_method(domain_coordinates, domain_values, training_data, eval_dataset, params, evaluation)



if what_to_test == 'chodci':
    # dvere, obecne nepouzivat, pripadne jen pri evaluation[0] = False a tim padem znamym poctu period
    ##################################################
    # where are the centers on the first two circles?
    #################################################
    #import matplotlib.pyplot as plt
    #import matplotlib.colors as colors
    #X = dio.create_X(training_data, structure, transformation)
    #plt.scatter(X[:, 0], X[:, 1], color='b')
    #plt.scatter(C[:,0], C[:,1], color='r')
    #plt.scatter(X[:, 2] * 1.3, X[:, 3] * 1.3, color='y')
    #plt.scatter(C[:,2] * 1.3, C[:,3] * 1.3, color='g')
    #plt.scatter(X[:, 4] * 1.7, X[:, 5] * 1.7, color='m')
    #plt.scatter(C[:,4] * 1.7, C[:,5] * 1.7, color='k')
    #plt.savefig("centres.png")
    #plt.close()
    # konec dveri
    pass


################
# estimation
###############
if what_to_test == 'chodci':
    # chodci
    testset = np.loadtxt('../data/wednesday_thursday_nights.txt')
    test_data, positive_data, eval_dataset_t= dio.get_data(testset)
    test_domain_coordinates, test_domain_values = gr.get_domain(test_data, positive_data, edges_of_cell, edges_of_big_cell)
    TESTING = dio.create_X(domain_coordinates, structure, transformation)
    PREDICTION = dio.create_X(test_domain_coordinates, structure, transformation)
    pred = es.training_model(PREDICTION, C, densities, COV, k, params, structure, heights)

if what_to_test == 'dvere':
    # dvere
    #dataset = np.c_[np.loadtxt('../data/test_times_9.txt'), np.loadtxt('../data/test_data_9.txt')]
    #dataset = np.c_[np.loadtxt('../data/test_times_8.txt'), np.loadtxt('../data/test_data_8.txt')]
    TESTING = dio.create_X(dataset[:, 0:1], structure, transformation)

if what_to_test == 'dvere_hodina':
    # hodinove agregaty !!!
    TESTING = dio.create_X(dataset[:, 0:1], structure, transformation)

out = es.training_model(TESTING, C, densities, COV, k, params, structure, heights)





################
# outputs
###############
if what_to_test == 'chodci':
    # chodci
    #print('statistiky odhadu')
    #print('prumer')
    #print(np.mean(out))
    #print('minimum')
    #print(np.min(out))
    #print('maximum')
    #print(np.max(out))

    #print('\n')
    #print('statistiky mereni')
    #print('prumer')
    #print(np.mean(domain_values))
    #print('minimum')
    #print(np.min(domain_values))
    #print('maximum')
    #print(np.max(domain_values))

    #print('\n')
    print('statistiky rozdilu')
    print('prumer')
    print(np.mean(np.abs(domain_values - out)))
    print('median')
    print(np.median(np.abs(domain_values - out)))
    print('minimum')
    print(np.min(np.abs(domain_values - out)))
    print('maximum')
    print(np.max(np.abs(domain_values - out)))
    print('prumerna vzdalenost modelu od mereni, RMSE')
    print(np.sqrt(np.mean((domain_values - out)**2)))
    #print('korelace mezi modelem a merenim')
    #print(np.corrcoef(domain_values.reshape(-1), out.reshape(-1)))
    #t_max = 1152
    #x_max = 200
    #y_max = 200
    #realita = np.histogramdd(domain_coordinates, bins=[t_max, x_max, y_max],
    #                                 range=None, normed=False, weights=domain_values.reshape(-1))[0]
    #model = np.histogramdd(domain_coordinates, bins=[t_max, x_max, y_max],
    #                                 range=None, normed=False, weights=out.reshape(-1))[0]
    #vi.model_visualisation(model, realita, np.shape(realita), 360.0, 'pokus_')
    #visualize_data = np.loadtxt('../data/wednesday_thursday_nights.txt')
    #vi.imgs_creation(visualize_data, C, densities, COV, k, params, structure, transformation, edges_of_cell)
    #test_dataset = np.loadtxt('../data/siroci_lide_test_600.0_0.1.txt')
    #all_test_data, training_test_data = dio.get_data(test_dataset)
    #test_domain_coordinates, test_domain_values = gr.get_domain(all_test_data, training_test_data, edges_of_cell, edges_of_big_cell)
    #np.savetxt('../data/siroci_lide_test_coordinates_600.0_0.1.txt', np.c_[test_domain_coordinates, test_domain_values])
    #test_TESTING = dio.create_X(test_domain_coordinates, structure, evaluation[3])
    #test_out = es.training_model(test_TESTING, C, densities, COV, k, params, structure)
    #test_rozdil = np.sqrt(np.sum((test_domain_values - test_out)**2))
    #print('vzdalenost modelu od testu')
    #print(test_rozdil)
    print('prumerna vzdalenost predikce od mereni, RMSE')
    print(np.sqrt(np.mean((test_domain_values - pred)**2)))
    print('prumerna vzdalenost nul od mereni, RMSE')
    print(np.sqrt(np.mean((test_domain_values - np.zeros_like(test_domain_values))**2)))
    print('statistiky rozdilu')
    print('prumer')
    print(np.mean(np.abs(test_domain_values - pred)))
    print('median')
    print(np.median(np.abs(test_domain_values - pred)))
    print('minimum')
    print(np.min(np.abs(test_domain_values - pred)))
    print('maximum')
    print(np.max(np.abs(test_domain_values - pred)))
    import matplotlib.pyplot as plt
    plt.plot(test_domain_values[:10000])
    plt.plot(pred[:10000], color='r')
    plt.savefig('srovnani_hodnot.png')
    plt.close()
    # konec chodcu


if what_to_test == 'dvere':
    # dvere
    #out = out / coeff
    X_test_values = dataset[:, 1]
    print('RMSE: ')
    print(np.sqrt(np.mean((out - X_test_values) ** 2)))
    print('MSE: ')
    print(np.mean((out - X_test_values) ** 2))
    import matplotlib.pyplot as plt
    #plt.plot(X_test_values[:10000], color='y')
    #plt.plot(out[:10000], color='g')
    plt.plot(X_test_values[168: 168*2], color='y')
    plt.plot(out[168: 168*2], color='g')
    #plt.plot(1 * (X_test_values[:10000] - out[:10000]) / (out[:10000] + 0.1), color='b', linestyle='None', marker=',')
    #plt.plot(1 * (moving_average(X_test_values[:10000] - out[:10000], n=60)) / (moving_average(out[:10000], n=60) + 0.1), color='r', linestyle='None', marker=',')
    #plt.plot((1.0/60) * (moving_sum((moving_average(X_test_values[:10000] - out[:10000], n=60)) / (moving_average(out[:10000], n=60) + 0.1), n=60)), color='r', linestyle='None', marker=',')
    #plt.plot((1.0) * ((moving_average(X_test_values[:10000] - out[:10000], n=60)) / (moving_average(out[:10000], n=60) + 0.1)), color='b', linestyle='None', marker=',')
    #plt.plot(1 * moving_sum(X_test_values[:10000] - out[:10000], n=20) / (moving_sum(out[:10000], n=20) + 0.1), color='r', linestyle='None', marker=',')
    #plt.plot(X_test_values[30:10000], color='y')
    #plt.plot(moving_average(X_test_values[:10030], n=60), color='g')
    #a = np.loadtxt('../data/testing_means_minutes.txt')[:, 1]
    #plt.plot(a[:10000], color='g')
    #print(np.mean((a - X_test_values) ** 2))
    #plt.plot(out[:10000], color='b')
    #plt.ylim(-0.5, 2)
    plt.savefig("../figs/temp.png")
    plt.close()
    # konec dveri

if what_to_test == 'dvere_hodina':
    # dvere
    X_test_values = dataset[:, 1]
    print(np.mean((out - X_test_values) ** 2))
    import matplotlib.pyplot as plt
    #plt.plot(X_test_values, color='y')
    #plt.plot(out, color='g')
    plt.plot(X_test_values[: 168], color='y')
    a = np.loadtxt('../data/testing_means.txt')[:, 1]
    plt.plot(a[:168], color='g')
    plt.plot(out[: 168], color='b')
    plt.ylim(-0.5, 80)
    plt.savefig("temp60.png")
    plt.close()
    print(np.mean((a - X_test_values) ** 2))
    # konec dveri


"""
#########################################################
# FIND WEIGHTS ON THE WHOLE TRAINING DATASET, aka "model"
# CALIBRATION
#########################################################

# v tuto chvili mam X, C, WEIGHTS, k
print(WEIGHTS)
U = cl.partition_matrix(WEIGHTS, 'hard')
COV = []
for cluster in xrange(k):
    weights = U[cluster]
    XminusC = bs.substraction(X, C[cluster], sid, structure)
    COV.append(np.cov(XminusC, bias=True, rowvar=False, aweights=weights))
COV = np.array(COV)
Pi = np.mean(U, axis=1)
# ted mam X, C, WEIGHTS, k, COV, U, Pi
# ale potrebuji jen C, COV, Pi

print('toto jse C')
print(C)
print('toto je COV')
print(COV)
print('toto je Pi')
print(Pi)

X_vse = dio.create_X(dataset[:, 0:1], structure, transformation)
DIST = []
for cluster in xrange(k):
    X_MU = bs.substraction(X_vse, C[cluster], sid, structure)
    #DIST.append(bs.transf_dist(X_MU, 'MVN', COV[cluster]))
    DIST.append(bs.transf_dist(X_MU, 'NORM_MODEL', COV[cluster]))
DIST = np.array(DIST)

# a ted prepocitam Pi
#Pi_new = Pi * np.sum(U, axis=1) / np.sum(DIST, axis=1)
Pi_new = np.sum(U, axis=1) / np.sum(DIST, axis=1)

# cimz ziskavam Pi_new, C, COV  ... a to je muj model

print('nove Pi')
print(Pi_new)    
#print('soucet DIST')
#print(np.sum(DIST))
#print(np.sum(DIST, axis=1))
#print(np.max(DIST, axis=1))
#out = (DIST).sum(axis=0)  
    


#######################
# WHOLE MODEL CREATED
# CALLING 'ESTIMATE'
#######################
print('a nyni jakoze estimate')
X_testovaci = dio.create_X(dataset[:, 0:1], structure, transformation)
#X_testovaci = X
DIST = []
for cluster in xrange(k):
    X_MU = bs.substraction(X_testovaci, C[cluster], sid, structure)
    #DIST.append(Pi_new[cluster] * bs.transf_dist(X_MU, 'MVN', COV[cluster]))
    DIST.append(Pi_new[cluster] * bs.transf_dist(X_MU, 'NORM_MODEL', COV[cluster]))
DIST = np.array(DIST)
#out = (DIST).sum(axis=0)  
out = (DIST).max(axis=0)  # for 'NORM_MODEL'

print('soucet DIST')
print(np.sum(DIST))
"""

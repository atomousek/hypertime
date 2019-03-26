
# Created on Sun Aug 27 14:40:40 2017
# @author: tom

"""
returns parameters of the learned model
call proposed_method(longest, shortest, path, edge_of_square, timestep, k,
                     radius, number_of_periods, evaluation)
where
input: longest float, legth of the longest wanted period in default
                      units
       shortest float, legth of the shortest wanted period
                       in default units
       path string, path to file
       edge_of_square float, spatial edge of cell in default units (meters)
       timestep float, time edge of cell in default units (seconds)
       k positive integer, number of clusters
       radius float, size of radius of the first found hypertime circle
       number_of_periods int, max number of added hypertime circles
       evaluation boolean, stop learning when the error starts to grow?
and
output: C numpy array kxd, matrix of k d-dimensional cluster centres
        COV numpy array kxdxd, matrix of covariance matrices
        density_integrals numpy array kx1, matrix of ratios between
                                           measurements and grid cells
                                           belonging to the clusters
        structure list(int, list(floats), list(floats)),
                  number of non-hypertime dimensions, list of hypertime
                  radii nad list of wavelengths
        average DODELAT
"""

import numpy as np
from time import clock
import copy as cp

import fremen as fm
import initialization as it
import grid as gr
import dataset_io as dio
import clustering as cl
#import calibration as ca
import basics as bs
import estimation as es
import gmm


def proposed_method(domain_coordinates, domain_values, training_data, eval_dataset, params, evaluation):
    """
    input: longest float, legth of the longest wanted period in default
                          units
           shortest float, legth of the shortest wanted period
                           in default units
           dataset numpy array, columns: time, vector of measurements, 0/1
                                (occurence of event)
           edge_of_square float, spatial edge of cell in default units (meters)
           timestep float, time edge of cell in default units (seconds)
           k positive integer, number of clusters
           radius float, size of radius of the first found hypertime circle
           number_of_periods int, max number of added hypertime circles
           evaluation boolean, stop learning when the error starts to grow?
    output: C numpy array kxd, matrix of k d-dimensional cluster centres
            COV numpy array kxdxd, matrix of covariance matrices
            density_integrals numpy array kx1, matrix of ratios between
                                               measurements and grid cells
                                               belonging to the clusters
            structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
            average DODELAT
    uses: time.clock()
          init.whole_initialization(), iteration_step()
    objective: to learn model parameters
    """
    if evaluation[0] == False:
        # for the future to know the strusture of evaluation
        edges_of_cell = evaluation[1]
        edges_of_big_cell = evaluation[2]
        transformation = evaluation[3]
        max_number_of_periods = evaluation[4]  # not used here
        longest, shortest = evaluation[5]  # not used here
        structure = evaluation[6]  # not used for evaluation[0] = True
        k = evaluation[7]  # not used for evaluation[0] = True

        X = dio.create_X(training_data, structure, transformation)
        DOMAIN = dio.create_X(domain_coordinates, structure, transformation)
        eval_domain = (dio.create_X(eval_dataset[0], structure, transformation), eval_dataset[1], dio.create_X(eval_dataset[2], structure, transformation), eval_dataset[3])
        diff, C, densities, COV, difference, heights = iteration_step(DOMAIN, domain_values, X, k, structure, params, eval_domain)
        #C, U = cl.iteration(X, k, structure, params)
        #densities, COV = ca.body(DOMAIN, X, C, U, k, params, structure)
    else:
        edges_of_cell = evaluation[1]
        edges_of_big_cell = evaluation[2]
        transformation = evaluation[3]
        max_number_of_periods = evaluation[4]
        longest, shortest = evaluation[5]
        k = evaluation[7]
        # initialization
        frequencies = it.build_frequencies(longest, shortest)
        structure = it.first_structure(training_data)
        if structure[0] == 0 and structure[1] == []:
            # there is nothing to cluster, we have to create new structure with one 'circle' before clustering
            average = domain_values / len(domain_values)
            #C = np.array([average])
            #COV = C/10
            #densities = np.array([[average]])
            #k = 1
            #chosen_period(T, S, W)
            the_period = fm.chosen_period(domain_coordinates[:, 0], domain_values - average, frequencies)[0]
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            structure[1].append(bs.radius(the_period, structure))
            structure[2].append(the_period)
            WW = list(frequencies)
            #print(1/the_period)
            WW.remove(1 / the_period)  # P
            frequencies = np.array(WW)
            print('nothing to cluster, periodicity ' + str(the_period) + ' chosen and the corresponding frequency removed')
        # create model
        diff, C, densities, COV, the_period, k, heights = best_diff(training_data, domain_coordinates, domain_values, frequencies, k, structure, params, transformation, eval_dataset)
        jump_out = 0
        iteration = 0
        #diff = -1
        while jump_out == 0:
            #print('\nstarting learning iteration: ' + str(iteration))
            #print('with number of clusters: ' + str(k))
            #print('and the structure: ' + str(structure))
            iteration += 1
            start = clock()
            jump_out, diff, C, densities, COV, the_period, structure, frequencies, k, heights = \
                step_evaluation(diff, C, densities, COV, the_period, structure, frequencies, training_data, domain_coordinates, domain_values, transformation, k, params, heights, eval_dataset)
            finish = clock()
            print('structure: ' + str(structure) + ', number of clusters: ' + str(k) + ', and difference to training data: ' + str(diff))
            #print('leaving learning iteration: ' + str(iteration))
            #print('processor time: ' + str(finish - start))
            if len(structure[1]) >= max_number_of_periods:
                jump_out = 1
        #print('learning iterations finished')
    #return C, densities, COV, k, params, structure  # to poradi pak budu muset zvazit ... proc vracim params????
    return C, densities, COV, k, structure, heights


def step_evaluation(diff, C, densities, COV, the_period, structure, frequencies, training_data, domain_coordinates, domain_values, transformation, k, params, heights, eval_dataset):
    """
    input: path string, path to file
           input_coordinates numpy array, coordinates for model creation
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           C numpy array kxd, centres from last iteration
           U numpy array kxn, matrix of weights from the last iteration
           k positive integer, number of clusters
           shape_of_grid numpy array dx1 int64, number of cells in every
                                                dimension
           time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                            over every
                                                            timeframe
           T numpy array shape_of_grid[0]x1, time positions of timeframes
           W numpy array Lx1, sequence of reasonable frequencies
           ES float64, squared sum of squares of residues from this iteration
           COV numpy array kxdxd, matrix of covariance matrices
           density_integrals numpy array kx1, matrix of ratios between
                                              measurements and grid cells
                                              belonging to the clusters
           P float64, length of the most influential frequency in default
                      units
           radius float, size of radius of the first found hypertime circle
    output: jump_out int, zero or one - to jump or not to jump out of learning
            structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
            C numpy array kxd, matrix of k d-dimensional cluster centres
            U numpy array kxn, matrix of weights
            COV numpy array kxdxd, matrix of covariance matrices
            density_integrals numpy array kx1, matrix of ratios between
                                               measurements and grid cells
                                               belonging to the clusters
            W numpy array Lx1, sequence of reasonable frequencies
            ES float64, squared sum of squares of residues from this iteration
            P float64, length of the most influential frequency in default
                       units
    uses: iteration_step()
          cp.deepcopy()
    objective: to send new or previous version of model (and finishing pattern)
    """
    new_structure = cp.deepcopy(structure)
    # new: structure, DOMAIN, X, frequencies (and also k and the_period)
    new_structure[1].append(bs.radius(the_period, new_structure))
    new_structure[2].append(the_period)
    WW = list(frequencies)
    WW.remove(1 / the_period)  # P
    new_frequencies = np.array(WW)
    #print('periodicity ' + str(the_period) + ' chosen and the corresponding frequency removed')
    ##########################################
    last_best = best_diff(training_data, domain_coordinates, domain_values, new_frequencies, k, new_structure, params, transformation, eval_dataset)
    ##########################################
    if last_best[0] < diff or diff == -1:  # (==) if diff < diff_old
        diff, C, densities, COV, the_period, k, heights = last_best
        structure = cp.deepcopy(new_structure)
        frequencies = np.empty_like(new_frequencies)
        np.copyto(frequencies, new_frequencies)
        jump_out = 0
    else:
        jump_out = 1
        print('error of model have risen, when structure ' + str(new_structure) + ' tested')
    return jump_out, diff, C, densities, COV, the_period, structure, frequencies, k, heights


def best_diff(training_data, domain_coordinates, domain_values, frequencies, k, new_structure, params, transformation, eval_dataset):
    """
    """
    X = dio.create_X(training_data, new_structure, transformation)
    DOMAIN = dio.create_X(domain_coordinates, new_structure, transformation)

    eval_domain = (dio.create_X(eval_dataset[0], new_structure, transformation), eval_dataset[1], dio.create_X(eval_dataset[2], new_structure, transformation), eval_dataset[3])

    list_of_others = []
    list_of_diffs = []
    list_of_differences = []

    #test_vals = []
    #test_times = []
    #for file in xrange(1, 10):
    #    test_vals.append(np.loadtxt('../data/test_data_' + str(file) + '.txt'))
    #    test_times.append(dio.create_X(np.loadtxt('../data/test_times_' + str(file) + '.txt').reshape(-1,1), new_structure, transformation))


    for j in xrange(21):  # for the case that the clustering would fail TRY TO DO IT ONLY ONCE BECAUSE OF NEW INITIALIZATION !!!
        diff_j, C_j, densities_j, COV_j, difference_j, heights_j =\
            iteration_step(DOMAIN, domain_values, X, k, new_structure, params, eval_domain)
        list_of_diffs.append(diff_j)
        list_of_others.append((diff_j, C_j, densities_j, COV_j, k, heights_j))
        list_of_differences.append(difference_j)

        #predictions = []
        #for q in xrange(9):
        #    out = es.training_model(test_times[q], C_j, densities_j, COV_j, k, params, new_structure, heights_j)
        #    vals = test_vals[q]
        #    predictions.append(np.mean((out - vals) ** 2))
        #
        #print(diff_j, fm.chosen_period(domain_coordinates[:, 0], difference_j, frequencies)[1], tuple(predictions))
    list_of_diffs = np.array(list_of_diffs)
    chosen_model = np.where(np.median(list_of_diffs))[0][0]  # find index of median difference between model and training data
    the_period, tested_sum_of_amplitudes = fm.chosen_period(domain_coordinates[:, 0], list_of_differences[chosen_model], frequencies)   # tested_sum_of_amplitudes not used in this version
    diff, C, densities, COV, k, heights = list_of_others[chosen_model]
    return diff, C, densities, COV, the_period, k, heights


def iteration_step(DOMAIN, domain_values, X, k, structure, params, eval_domain):
    """
    """
    #C, densities, COV, heights = try_model(DOMAIN, X, k, structure, params, 0)
    C, U, COV, Pi = gmm.iteration(X, k, structure, params)
    densities, heights = gmm.calibration(DOMAIN, C, U, COV, Pi, k, params, structure)       
    domain_values_estimation = es.training_model(DOMAIN, C, densities, COV, k, params, structure, heights)
    difference = domain_values - domain_values_estimation

    eval_one_est = es.training_model(eval_domain[0], C, densities, COV, k, params, structure, heights)
    eval_two_est = es.training_model(eval_domain[2], C, densities, COV, k, params, structure, heights)
    diff = np.max((np.mean((eval_domain[1] - eval_one_est) ** 2), np.mean((eval_domain[3] - eval_two_est) ** 2)))


    #diff = np.mean(difference ** 2)
    return diff, C, densities, COV, difference, heights


def try_model(DOMAIN, X, k, structure, params, counter):
    if counter > 0:
        print('new try to create model: ' + str(counter))
    try:
        C, U, COV, Pi = gmm.iteration(X, k, structure, params)
        densities, heights = gmm.calibration(DOMAIN, C, U, COV, Pi, k, params, structure)       
        return C, densities, COV, heights
    except:
        return try_model(DOMAIN, X, k, structure, params, counter+1)

"""
def try_model(DOMAIN, X, k, structure, params, counter):
    if counter > 0:
        print('new try to create model: ' + str(counter))
    try:
        if params[1] != 'gmm':
            C, U = cl.iteration(X, k, structure, params)
            densities, COV = ca.body(DOMAIN, X, C, U, k, params, structure)
        else:
            C, U, COV, Pi = gmm.iteration(X, k, structure, params)
            densities, heights = gmm.calibration(DOMAIN, C, U, COV, Pi, k, params, structure)       
        return C, densities, COV, heights
    except:
        return try_model(DOMAIN, X, k, structure, params, counter+1)
"""








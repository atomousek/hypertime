# Created on Mon Jul 17 15:14:57 2017
# @author: tom

"""
creates grid above data and outputs central positions of cels of grid
(input_coordinates), number of cells in every dimension (shape_of_grid),
time positions based on the grid (T), numbers of measurements in
timeframes (time_frame_sums) and number of measurements in all dataset
(overall_sum).
call time_space_positions(edge_of_square, timestep, path)
where
input: edge_of_square float, spatial edge of cell in default units (meters)
       timestep float, time edge of cell in default units (seconds)
       path string, path to file
and
output: input_coordinates numpy array, coordinates for model creation
        time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                        over every
                                                        timeframe
        overall_sum number (np.float64 or np.int64), sum of all measures
        shape_of_grid numpy array dx1 int64, number of cells in every
                                             dimension
        T numpy array shape_of_grid[0]x1, time positions of timeframes

timestep and edge_of_square has to be chosen based on desired granularity,
timestep refers to the time variable,
edge_of_square refers to other variables - it is supposed that the step
    (edge of cell) in every variable other than time is equal.
    If there are no other variables, some value has to be added but it is not
    used.
"""

import numpy as np


def get_domain(dataset, edges_of_cell, edges_of_big_cell):
    """
    input: dataset ... numpy array, timestemp, position in space, measured values
           edges_of_cell ... list, lengths of individual edges of smallest cell
           edges_of_big_cell ... list, lower resolution cells (creates surroundings of measured data)
    output: domain_coordinates ... numpy array, "list" of coordinates of cells were some measurement was performed
            #domain_frequencies ... numpy array, number of measurements in each cell
            domain_sums ... numpy array, sums of measured values
    """
    extended_shape_of_grid, uniform_histogram_bools = get_uniform_data(dataset[:, 0: -1], edges_of_cell, edges_of_big_cell)
    domain_coordinates = get_coordinates(dataset[:, 0: -1], extended_shape_of_grid, uniform_histogram_bools)
    # domain_values jsou cetnosti - puvodni kod ocekaval hodnotu 1 pro kazde mereni
    #domain_frequencies = get_frequencies(dataset[:, 0: -1], extended_shape_of_grid, uniform_histogram_bools)
    domain_sums = get_sums(dataset, extended_shape_of_grid, uniform_histogram_bools)
    return np.float64(domain_coordinates), domain_sums
    
    
def get_occupied_coor(data, edges_of_big_cell):
    """
    """
    shape_of_big_grid = number_of_cells(data, edges_of_big_cell)
    big_histogram, big_edges = np.histogramdd(data, bins=shape_of_big_grid[0],
                                      range=shape_of_big_grid[1],
                                      normed=False, weights=None)
    big_central_points = []
    for i in range(len(big_edges)):
        step_lenght = (big_edges[i][-1] - big_edges[i][0]) / (len(big_edges[i] - 1))
        big_central_points.append(big_edges[i][0: -1] + step_lenght / 2)
    big_coordinates = cartesian_product(*big_central_points)
    big_histogram_values = big_histogram.reshape(-1)
    occupied_coordinates = big_coordinates[big_histogram_values > 0]
    return occupied_coordinates

def get_uniform_data(data, edges_of_cell, edges_of_big_cell):
    """
    """
    # find coordinates of occupied big cells
    occupied_coordinates = get_occupied_coor(data, edges_of_big_cell)
    edges_of_cell_af = np.array(edges_of_cell, dtype=float)
    edges_rates = np.floor(np.array(edges_of_big_cell, dtype=float) / edges_of_cell_af) + 1.0
    sequences = []
    rate_of_new_points = 1
    for j in range(len(edges_of_cell)):
        sequence = np.arange(edges_rates[j]) * edges_of_cell_af[j]
        sequences.append(sequence - np.mean(sequence))
        rate_of_new_points *= edges_rates[j]
    rate_of_new_points = np.int64(rate_of_new_points)
    uniform_data = np.empty((len(occupied_coordinates) * rate_of_new_points, len(edges_of_cell)))
    counter = 0
    for coordinate in occupied_coordinates:
        uniform_points = []
        for k in range(len(edges_of_cell)):
            uniform_points.append(coordinate[k] + sequences[k])
        uniform_data[counter * rate_of_new_points: (counter + 1) * rate_of_new_points, :] = cartesian_product(*uniform_points)
        counter += 1
    # histogram on domain
    extended_shape_of_grid = number_of_cells(uniform_data, edges_of_cell)
    uniform_histogram = np.histogramdd(uniform_data, bins=extended_shape_of_grid[0],
                                      range=extended_shape_of_grid[1],
                                      normed=False, weights=None)[0]
    uniform_histogram_bools = (uniform_histogram.reshape(-1) > 0)
    return extended_shape_of_grid, uniform_histogram_bools


def get_coordinates(data, extended_shape_of_grid, uniform_histogram_bools):
    """
    """
    edges = np.histogramdd(data, bins=extended_shape_of_grid[0],
                                      range=extended_shape_of_grid[1],
                                      normed=False, weights=None)[1]
    central_points = []
    for i in range(len(edges)):
        step_lenght = (edges[i][-1] - edges[i][0]) / (len(edges[i] - 1))
        central_points.append(np.array(edges[i][0: -1] + step_lenght / 2, dtype=np.float32))
    coordinates = cartesian_product(*central_points)
    domain_coordinates = coordinates[uniform_histogram_bools]
    return domain_coordinates


def get_frequencies(data, extended_shape_of_grid, uniform_histogram_bools):
    """
    """
    histogram = np.histogramdd(data, bins=extended_shape_of_grid[0],
                                      range=extended_shape_of_grid[1],
                                      normed=False, weights=None)[0]
    histogram_freqs = histogram.reshape(-1)
    domain_frequencies = histogram_freqs[uniform_histogram_bools]
    return domain_frequencies


def get_sums(dataset, extended_shape_of_grid, uniform_histogram_bools):
    """
    """
    histogram = np.histogramdd(dataset[:, 0: -1], bins=extended_shape_of_grid[0],
                                      range=extended_shape_of_grid[1],
                                      normed=False, weights=dataset[:, -1])[0]
    histogram_sums = histogram.reshape(-1)
    domain_sums = histogram_sums[uniform_histogram_bools]
    return domain_sums


def number_of_cells(data, edges_of_cell):
    """
    input: X numpy array nxd, matrix of measures
           edge_of_square float, length of the edge of 2D part of a "cell"
           timestep float, length of the time edge of a "cell"
    output: shape_of_grid numpy array, number of edges on t, x, y, ... axis
    uses:np.shape(), np.max(), np.min(),np.ceil(), np.int64()
    objective: find out number of cells in every dimension
    """
    # number of predefined cubes in the measured space
    # changed to exact length of timestep and edge of square
    # changed to general shape of cell
    extended_shape_of_grid = [[],[]]
    n, d = np.shape(data)
    for i in range(d):
        min_i = np.min(data[:, i])
        max_i = np.max(data[:, i])
        range_i = max_i - min_i
        edge_i = float(edges_of_cell[i])
        number_of_bins = np.floor(range_i / edge_i) + 1
        half_residue = (edge_i - (range_i % edge_i)) / 2.0
        position_min =  min_i - half_residue
        position_max =  max_i + half_residue
        extended_shape_of_grid[0].append(int(number_of_bins))
        extended_shape_of_grid[1].append([position_min, position_max])
    return extended_shape_of_grid


def cartesian_product(*arrays):
    """
    downloaded from:
    'https://stackoverflow.com/questions/11144513/numpy-cartesian-product-of'+\
    '-x-and-y-array-points-into-single-array-of-2d-points'
    input: *arrays enumeration of central_points
    output: numpy array (central positions of cels of grid)
    uses: np.empty(),np.ix_(), np.reshape()
    objective: to perform cartesian product of values in columns
    """
    la = len(arrays)
    arr = np.empty([len(a) for a in arrays] + [la],
                   dtype=arrays[0].dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)

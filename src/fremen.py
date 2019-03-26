"""
basic FreMEn to find most influential periodicity, call
chosen_period(T, time_frame_sums, time_frame_freqs, longest, shortest, W, ES):
it returns the most influential period in the timeseries, where timeseries are
    the residues between reality and model
where
input: T numpy array Nx1, time positions of measured values
       time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                        over every
                                                        timeframe
       time_frame_freqs numpy array shape_of_grid[0]x1, sum of
                                                        frequencies(stat)
                                                        in model
                                                        over every
                                                        timeframe
       W numpy array Lx1, sequence of reasonable frequencies
       ES float64, squared sum of squares of residues from the last
                   iteration
and
output: P float64, length of the most influential frequency in default
        units
        W numpy array Lx1, sequence of reasonable frequencies without
                           the chosen one
        ES_new float64, squared sum of squares of residues from
                        this iteration
        dES float64, difference between last and new error

for the creation of a list of reasonable frequencies call
build_frequencies(longest, shortest):
where
longest - float, legth of the longest wanted period in default units,
        - usualy four weeks
shortest - float, legth of the shortest wanted period in default units,
         - usualy one hour.
It is necessary to understand what periodicities you are looking for (or what
    periodicities you think are the most influential)
"""

import numpy as np
from collections import defaultdict


def chosen_period(T, S, W):
    """
    input: T numpy array Nx1, time positions of measured values
           time_frame_sums numpy array shape_of_grid[0]x1, sum of measures
                                                            over every
                                                            timeframe
           time_frame_freqs numpy array shape_of_grid[0]x1, sum of
                                                            frequencies(stat)
                                                            over every
                                                            timeframe
           W numpy array Lx1, sequence of reasonable frequencies
           ES float64, squared sum of squares of residues from the last
                       iteration
    output: P float64, length of the most influential frequency in default
            units
            W numpy array Lx1, sequence of reasonable frequencies without
                               the chosen one
            ES_new float64, squared sum of squares of residues from
                            this iteration
            dES float64, difference between last and new error
    uses: np.sum(), np.max(), np.absolute()
          complex_numbers_batch(), max_influence()
    objective: to choose the most influencing period in the timeseries, where
               timeseries are the residues between reality and model
    """
    # originally: S = (time_frame_sums - time_frame_freqs)[valid_timesteps]
    G = complex_numbers_batch(T, S, W)
    P = max_influence(W, G)
    # power spectral density ???
    sum_of_amplitudes =  np.sum(np.absolute(G) ** 2)
    #sum_of_amplitudes = np.sum(np.absolute(G))
    return P, sum_of_amplitudes


def complex_numbers_batch(T, S, W):
    """
    input: T numpy array Nx1, time positions of measured values
           S numpy array Nx1, sequence of measured values
           W numpy array Lx1, sequence of reasonable frequencies
    output: G numpy array Lx1, sequence of complex numbers corresponding
            to the frequencies from W
    uses: np.e, np.newaxis, np.pi, np.mean()
    objective: to find sparse(?) frequency spectrum of the sequence S
    """
    #Gs = S * (np.e ** (W[:, np.newaxis] * T * (-1j) * np.pi * 2))
    #print('mam Gs')
    #G = np.mean(Gs, axis=1)
    t, s = sum_by_times(T, S)
    G = []
    for i in xrange(len(W)):
        Gs = s * (np.e ** (W[i] * t * (-1j) * np.pi * 2))
        G.append(np.mean(Gs))
    G = np.array(G)
    return G


def max_influence(W, G):
    """
    input: W numpy array Lx1, sequence of reasonable frequencies
           G numpy array Lx1, sequence of complex numbers corresponding
                              to the frequencies from W
    output: P float64, length of the most influential frequency in default
                       units
            W numpy array Lx1, sequence of reasonable frequencies without
                               the chosen one
    uses: np.absolute(), np.argmax(), np.float64(),np.array()
    objective: to find length of the most influential periodicity in default
               units and return changed list of frequencies
    """
    maximum_position = np.argmax(np.absolute(G[1:])) + 1
    influential_frequency = W[maximum_position]
    # not sure if it is necessary now
    if influential_frequency == 0 or np.isnan(np.max(np.absolute(G))):
        print('problems in fremen.max_influence')
        P = np.float64(0.0)
    else:
        P = 1 / influential_frequency
    return P


def sum_by_times(T, S):
    """
    """
    TS = np.c_[T, S]
    ts_dict = defaultdict(float)
    for t, s in TS:
        ts_dict[t] += s
    ts_array = np.array(ts_dict.items())
    return ts_array[:, 0], ts_array[:, 1]


def build_frequencies(longest, shortest):  # should be part of initialization of learning
    """
    input: longest float, legth of the longest wanted period in default
                          units
           shortest float, legth of the shortest wanted period
                           in default units
    output: W numpy array Lx1, sequence of frequencies
    uses: np.arange()
    objective: to find frequencies w_0 to w_k
    """
    k = int(longest / shortest) + 1
    W = np.float64(np.arange(k)) / float(longest)
    return W



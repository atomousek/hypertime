import numpy as np
import basics as bs
import clustering as cl

def first_structure(training_data):
    """
    objective: to create initial structure
    todo: what anout different structure of structure, better for save and load to 'c++'
    """
    dim = np.shape(training_data)[1] - 1
    structure = [dim, [], []]
    return structure


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


def initialization(X, k, structure, params):
    """
    """
    n, d = np.shape(X)
    C = np.empty((k, d))
    WEIGHTS = np.empty((k, n))
    
    # pokusim se udelat ten suj algoritmus
    Lambda = 5
    Clambda = np.empty((k * Lambda, d))
    for redundant in range(k * Lambda):
        Clambda[redundant] = X[np.random.choice(n)]
    DIFFERENCES = []
    for point in Clambda:
        row = bs.substraction(Clambda, point, 'def', structure)
        DIFFERENCES.append(row)
    DIFFERENCES = np.array(DIFFERENCES)
    DISTANCES = np.sqrt(np.sum((DIFFERENCES)**2, axis=2))
    bigConstant = np.max(DISTANCES) * 2
    SEARCH = DISTANCES + np.eye(k * Lambda) * bigConstant
    sums = np.sum(DISTANCES, axis=0)
    for iteration in range(k * (Lambda - 1)):
        ind = np.unravel_index(np.argmin(SEARCH, axis=None), SEARCH.shape)
        toDel = np.argmin(sums[list(ind)])
        SEARCH[:, ind[toDel]] = bigConstant
        SEARCH[ind[toDel], :] = bigConstant
        sums[ind[toDel]] = bigConstant
    C = Clambda[sums != bigConstant]
    for cluster in range(k):
        XminusC = bs.substraction(X, C[cluster], 'def', structure)
        WEIGHTS[cluster] = cl.weights_matrix(XminusC, weights=None, uid='hard')
    U = cl.partition_matrix(WEIGHTS, uid='hard')
    """
    # PUVODNI A FUNKCNI
    # initialize M to random, initialize C to spherical with variance 1
    for cluster in xrange(k):
        C[cluster] = X[np.random.choice(n)]
        XminusC = bs.substraction(X, C[cluster], 'def', structure)
        WEIGHTS[cluster] = weights_matrix(XminusC, weights=None, uid='hard')
    U = partition_matrix(WEIGHTS, uid='hard')
    # KONEC PUVODNIHO A FUNKCNIHO
    """
    return C, U

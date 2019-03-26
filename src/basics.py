import numpy as np
from scipy.stats import multivariate_normal
import scipy.linalg as sln

def substraction(X, Y, sid, structure = None):
    """
    input: X numpy array {1, n}xd, measured data
           Y numpy array 1xd, some point in the space
           sid string, type of the substraction ('def', 'cos')
           structure list (optional)
    output: XY numpy array nx1, distances between every X and Y
    objective: to find difference vector between a set of vectors and one vector
    uses: np.shape(), np.tile(), np.ones(), np.empty(), np.sum(), np.minimum(),
          np.maximum(), np.arccos()
    """
    # expanding Y to nxd array
    shp = np.shape(X)
    if len(shp) == 1:  # one dimensional array of X
        nY = Y
    else:
        n = shp[0]
        nY = np.tile(Y, (n, 1))
    # calculating the distance
    if sid == 'def':
        # default X - Y
        XY = X - nY
    elif sid == 'cos' and structure != None:
        # hypertime dimensions substraction
        observations = np.shape(X)[0]
        ones = np.ones((observations, 1))
        dim = structure[0]
        radii = structure[1]
        XY = np.empty((observations, dim + len(radii)))
        XY[:, : dim] = X[:, : dim] - nY[:, : dim]
        for period in range(len(radii)):
            r = radii[period]
            cos = (np.sum(X[:, dim + (period * 2): dim + (period * 2) + 2] *
                          nY[:, dim + (period * 2): dim + (period * 2) + 2],
                          axis=1, keepdims=True) /
                         (r ** 2)
                  )
            cos = np.minimum(np.maximum(cos, ones * -1), ones)
            XY[:, dim + period: dim + period + 1] = r * np.arccos(cos) * np.sign(cos)
    elif sid == 'cos-def' and structure != None:
        # hypertime dimensions substraction with free position of cluster centres
        observations = np.shape(X)[0]
        ones = np.ones((observations, 1))
        dim = structure[0]
        radii = structure[1]
        XY = np.empty((observations, dim + len(radii)))
        XY[:, : dim] = X[:, : dim] - nY[:, : dim]
        for period in range(len(radii)):
            r = radii[period]
            cos = (np.sum(X[:, dim + (period * 2): dim + (period * 2) + 2] *
                          nY[:, dim + (period * 2): dim + (period * 2) + 2],
                          axis=1, keepdims=True) /
                         (r *  # suppose, every measuring is places on the circle
                          np.sqrt(np.sum(nY[:, dim + (period * 2): dim + (period * 2) + 2] ** 2, axis=1, keepdims=True))
                         )
                  )
            cos = np.minimum(np.maximum(cos, ones * -1), ones)
            XY[:, dim + period: dim + period + 1] = r * np.arccos(cos) * np.sign(cos)
    else:
        print('unknown substraction type, returning default')
        XY = X - nY
    return XY


def averaging(X, weights, structure, sid):
    """
    """
    if sid == 'def' or sid == 'cos-def':
        avg = np.dot(weights, X) / np.sum(weights)
    elif sid == 'cos':
        avg = np.dot(weights, X) / np.sum(weights)
        dim = structure[0]
        radii = structure[1]
        for period in range(len(radii)):
            r = radii[period]
            avg[dim + (period * 2): dim + (period * 2) + 2] = avg[dim + (period * 2): dim + (period * 2) + 2] / np.sqrt(np.sum(avg[dim + (period * 2): dim + (period * 2) + 2] ** 2))
    else:
        print('unknown type of averaging, returning default')
        avg = np.dot(weights, X) / np.sum(weights)
    return avg
        
        


def distance(D, did, P = None):
    """
    input: D numpy array nxd, matrix of differences between 1xd vectors
           did string, type of the distance ('E', 'M', 'CH', 'P')
           P numpy array dxd, matice prechodu
    output: DIST numpy array nx1, distances between D and origin
    objective: to find distance between D and 0
    uses: np.shape(), np.tile(), np.sum(), np.sqrt(), np.abs(), np.max()
    """
    # calculating the distance
    if did == 'E':
        # Euclidean distance
        DIST = np.sqrt(np.sum((D)**2, axis=1))
    elif did == 'M':
        # Manhattan distance
        DIST = np.sum(np.abs(D), axis=1)
    elif did == 'CH':
        # Chebyshev distance
        DIST = np.max(np.abs(D), axis=1)
    elif did == 'P':
        # distance in diferent basis
        DIST = np.sum(np.dot(D, P) * D, axis=1)
    else:
        print('unknown distance, returning Euclidean distance')
        DIST = np.sqrt(np.sum((D)**2, axis=1))
    return DIST


def projection(data, structure, pid):
    """
    input: data numpy array nxd*, matrix of measures IRL, where d* is number
                                  of measured variables
           structure list(int, list(floats), list(floats)),
                      number of non-hypertime dimensions, list of hypertime
                      radii nad list of wavelengths
           pid string, type of projection ('C')
    output: X numpy array nxd, matrix of measures in hypertime
    uses: np.empty(), np.c_[]
    objective: to create X as a data in hypertime, where the structure
               of a space is derived from the varibale structure
    """
    if pid == 'C':    
        # projection: for every period one circe
        dim = structure[0]
        radii = structure[1]
        wavelengths = structure[2]
        X = np.empty((len(data), dim + len(radii) * 2))
        X[:, : dim] = data[:, 1: dim + 1]
        for period in range(len(radii)):
            r = radii[period]
            Lambda = wavelengths[period]
            X[:, dim: dim + 2] = np.c_[r * np.cos(data[:, 0] * 2 * np.pi / Lambda),
                                       r * np.sin(data[:, 0] * 2 * np.pi / Lambda)]
            dim = dim + 2
    else:
        print('unknown projection, returning circle for every period')
        X = projection(data, structure, pid='C')
    return X


def transf_dist(D, tid, COV=None):
    """
    """
    shp = np.shape(D)
    if len(shp) == 1:  # one dimensional array of D
        d = len(D)
    else:
        d = shp[1]
    if tid == 'E':
        # Euklides
        tDist = distance(D, 'E')
    elif tid == 'MVN':
        # multivariate normal
        tDist = multivariate_normal.pdf(D, np.zeros(d), COV, allow_singular=True)
    elif tid == 'GK':
        # Gustafson-Kessel
        Det = np.linalg.det(COV)
        if Det < 0:
            print('negative determinant')
            Det = np.abs(DET)
        if Det == 0:
            print('zero determinant')
            Det = 1e-15
        Detd = Det ** (1 / d)
        COVDet = COV / Detd
        try:
            GK = np.linalg.inv(COVDet)
        except:
            print(sys.exc_info()[0])
            print('inversion not possible, using pseudoinversion')
            GK = np.linalg.pinv(COVDet)
        tDist = distance(D, did = 'P', P = GK)
    elif tid == 'NORM':
        NORM = np.linalg.inv(sln.sqrtm(COV))
        tDist = distance(D, did = 'P', P = NORM)
    elif tid == 'NORM_MODEL':
        tail_shaper = 1.0  # cim vyssi, tim jsou ty "ocasy" strmejsi
        sigma_multiplier = 2.0  # do kolika "sigma" se to povazuje za rovnomerne 
        NORM = np.linalg.inv(sln.sqrtm(COV))
        DIST = np.sqrt(np.sum(np.dot(D, NORM) * D, axis=1))
        W_part = 1 / (DIST + np.exp(-100))
        W_part = W_part ** tail_shaper
        tDist = np.empty_like(W_part)
        np.copyto(tDist, W_part)
        tDist[W_part > ((1/sigma_multiplier)**tail_shaper)] = (1/sigma_multiplier)**tail_shaper
        tDist[W_part < ((1/sigma_multiplier)**tail_shaper)] = 0  # bez ocasu
    elif tid == 'MVN_N':
        # multivariate normal normalized
        print('determinant kovariancni matice je')
        print(np.linalg.det(COV))
        print('na 1/d je ')
        print((np.linalg.det(COV)) ** (1.0/d))
        print('kde d je')
        print(d)
        print((np.linalg.det(COV)) ** 0.5)
        NORM = np.linalg.inv(sln.sqrtm(COV))
        DIST = np.dot(D, NORM)  # pravdepodobne spatne, viz. 'NORM_MODEL'
        tDist = multivariate_normal.pdf(DIST, np.zeros(d), np.eye(d), allow_singular=True) #/ ((np.linalg.det(COV)) ** (1.0/d))
    else:
        print('unknown transformation, returning Euklides distances')
        tDist = distance(D, 'E')
    return tDist


def radius(the_period, structure):
    """
    probably in the future we can experiment ;)
    """
    return 1.0

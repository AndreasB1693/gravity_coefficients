import numpy as np
from numpy.linalg import norm
from numba import jit, prange


@jit('float64[::1](float64[::1], float64[:, ::1], float64[::1])', 
     nopython=True, nogil=True, cache=True, fastmath=True)
def propergation(pos, pos_3rd, GMs):
    """
    Calculate the acceleration due to the gravitation of third bodies, 
    which are assumed to be point masses located at the positions 
    specified in `pos_3rd`.

    Parameters
    ----------
    pos : numpy.ndarray, shape (3,), dtype float64
        Position vector in the inertial frame.
    pos_3rd : numpy.ndarray
        Array of the position vectors of all considered third bodies.
    GMs : numpy.ndarray
        Gravitational coefficients of the considered third bodies.

    Note
    ----
    This function is specifically designed for use in the integration 
    process and is optimized with Numba for improved performance.
        
    Returns
    -------
    acc : numpy.ndarray, shape (3,), dtype float64
        Acceleration vector in km/s^2.
    """
    
    acc = np.zeros(3, dtype=np.float64)
    for i, GM in enumerate(GMs):
        dist = pos - pos_3rd[i]
        acc += -GM *(dist/norm(dist)**3  + pos_3rd[i]/norm(pos_3rd[i])**3)
        
    return acc


def evaluation(pos, pos_3rd, GM):
    """
    Calculate the acceleration(s) due to the gravity of a third
    body, which is assumed to be a point mass located at the
    position(s) specified in `pos_3rd`.

    Parameters
    ----------
    pos : numpy.ndarray
        Position vector(s) in the inertial frame. If `pos` is 
        a 1-dimensional array, it represents a single position vector; 
        if it is 2-dimensional, each row represents a position vector.
    pos_3rd : numpy.ndarray
        Position vector(s) of the considered third body in the inertial 
        frame. If `pos_3rd` is a 1-dimensional array, it represents a 
        single position vector; if it is 2-dimensional, each row 
        represents a position vector.
    GM : numpy.ndarray
        Gravitational coefficients of the considered third body.

    Returns
    -------
    acc : numpy.ndarray
        Acceleration vector(s) in km/s^2. 
        The shape of the output matches the shape of `pos` or `pos_3rd`.
        
    Note
    ----
    The function automatically detects if the acceleration vector shall 
    be computed for one or multiple positions:
    """
    
    DIM_pos = pos.ndim 
    DIM_3rd = pos_3rd.ndim
    
    if DIM_pos == 1 and DIM_3rd == 1:
        dist = norm(pos - pos_3rd)
        dist_3rd = norm(pos_3rd)
        return -GM *((pos - pos_3rd)/dist**3  + pos_3rd/dist_3rd**3)
    
    elif DIM_pos == 2 and DIM_3rd == 1:
        dist = norm(pos - pos_3rd, axis=1, keepdims=True)
        dist_3rd = norm(pos_3rd)   
        return -GM *((pos - pos_3rd)/dist**3  + pos_3rd/dist_3rd**3)
       
    elif DIM_pos == 1 and DIM_3rd == 2:
        dist = norm(pos - pos_3rd, axis=1, keepdims=True)
        dist_3rd = norm(pos_3rd, axis=1, keepdims=True)
        return -GM *((pos - pos_3rd)/dist**3  + pos_3rd/dist_3rd**3)

    elif pos.shape == pos_3rd.shape:
        dist = norm(pos - pos_3rd, axis=1, keepdims=True)
        dist_3rd = norm(pos_3rd, axis=1, keepdims=True)
        return -GM *((pos - pos_3rd)/dist**3  + pos_3rd/dist_3rd**3)
    
    else:
        raise ValueError("pos or pos_3rd have the wrong shape!")


def third_bodies(pos, pos_3rd, GM):
    """
    Computes the sum of all accelerations due to the gravitational 
    influence of third bodies given in the dictonary `pos_3rd`.

    Parameters
    ----------
    pos : numpy.ndarray, shape (3,), dtype float64
        Position vector [km] of the primary body.
    pos_3rd : dict
        Dictionary containing the position vectors [km] of all 
        considered third bodies. The keys are the names or identifiers 
        of the third bodies, and the values are the corresponding 
        position vectors.
    GM : dict
        Dictionary containing the gravitational coefficients of the 
        considered third bodies. The keys are the names or identifiers 
        of the third bodies, and the values are the corresponding 
        standard gravitational parameter [km^3 s^2]. 

    Returns
    -------
    acc : numpy.ndarray, shape (3,), dtype float64
        Acceleration vector [km/s^2] experienced by the primary body due to the gravitational
        influence of the third bodies.

    Note:
    -----
    Computes the acceleration due to the gravitational influence of 
    third bodies, which are assumed to be point masses located at 
    the positions specified in `pos_3rd`. This function is for the 
    integration process of a ODE and is optimized with Numba for 
    improved performance.

        
    """

    pos = np.ascontiguousarray(pos)
    pos_3rd = np.ascontiguousarray(list(pos_3rd.values()))
    GM = np.ascontiguousarray(list(GM.values()))
    return propergation(pos, pos_3rd, GM)


def third_body(pos, pos_3rd, GM):
    """
    Calculate the acceleration(s) due to the gravity of a single third
    body.

    Parameters
    ----------
    pos : numpy.ndarray
        Position vector(s) in the inertial frame. If `pos` is 
        a 1-dimensional array, it represents a single position vector; 
        if it is 2-dimensional, each row represents a position vector.
    pos_3rd : numpy.ndarray
        Position vector(s) of the considered third body in the inertial 
        frame. If `pos_3rd` is a 1-dimensional array, it represents a 
        single position vector; if it is 2-dimensional, each row 
        represents a position vector.
    GM : numpy.ndarray
        Standard gravitational parameter [km^3 s^2] 
        of the considered third body.

    Returns
    -------
    acc : numpy.ndarray
        Acceleration vector(s) in km/s^2. 
        The shape of the output matches the shape of `pos` or `pos_3rd`.
        
    Note
    ----
    Compute the acceleration(s) due to the gravity of a third
    body, which is assumed to be a point mass located at the
    position(s) specified in `pos_3rd`. Note that all objects must 
    be considered in the same frame. The function automatically 
    detects if the acceleration vector shall be computed for one or 
    multiple positions:
    - pos=1d and pos_3rd=1d: point is fixed, 3rd body is fixed.
    - pos=2d and pos_3rd=1d: point is moving, 3rd body is fixed.
    - pos=1d and pos_3rd=2d: point is fixed, 3rd body is moving.
    - pos=2d and pos_3rd=2d: point is moving, 3rd body is moving.
    """
    return evaluation(pos, pos_3rd, GM)

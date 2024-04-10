import numpy as np
from numpy.linalg import norm
from numba import jit

@jit('float64[::1](float64[::1], float64)', 
     nopython=True, nogil=True, cache=True, fastmath=True)
def propergation(state, GM):
    """
    Compute the acceleration due to relativistic effects.

    Parameters
    ----------
    state : numpy.ndarray, shape (6,), dtype float64
        State vector of the satellite, including position
        and velocity [km, km/s].
    GM : float
        standard gravitational parameter [km^3 s^2].

    Returns
    -------
    numpy.ndarray, shape (3,), dtype float64
        Acceleration vector in km/s^2.
    
    Note
    ----
    This function is specifically designed for use in the integration 
    process and is optimized with Numba for improved performance.
    """
    
    LIGHTSPEED = 299792.458      
    pos = state[:3]
    vel = state[3:]
    
    
    r_abs = norm(pos)
    v_abs = norm(vel)
    fac = GM / (LIGHTSPEED**2.0 * r_abs**3.0) 
    acc_1 = (4.0 * GM / r_abs - v_abs**2.0) * pos 
    acc_2 = 4.0 * np.dot(pos, vel) * vel
    return fac*(acc_1 + acc_2)


def relativistic(state, GM):
    """
    Compute the acceleration(s) due to relativistic effects.

    Parameters
    ----------
    state : numpy.ndarray, shape (6,), dtype float64
        State vector(s) of the satellite, including position
        and velocity [km, km/s]. If `state` is a 1-dimensional array, 
        it represents a single state vector; if it is 2-dimensional, 
        each row represents a state vector.
    GM : float
        standard gravitational parameter [km^3 s^2].

    Returns
    -------
    numpy.ndarray, shape (3,), dtype float64
        Acceleration vector in km/s^2.
        The shape of the output matches the shape of `state`.
    
    Note
    ----
    Compute the acceleration(s) due to relativistic effects according
    to the post-Newtonian correction terms. This function automatically 
    detects if the acceleration vector shall be computed for one or
    multiple positions: 
    """
    if state.ndim == 1:
        return propergation(np.ascontiguousarray(state), GM)

    elif state.ndim == 2:
        LIGHTSPEED = 299792.458      
        pos = state[:, :3]
        vel = state[:, 3:]
        r_abs = norm(pos, axis=1, keepdims=True)
        v_abs = norm(vel, axis=1, keepdims=True)

        fac = GM / (LIGHTSPEED**2.0 * r_abs**3.0) 
        acc_1 = (4.0 * GM / r_abs - v_abs**2.0) * pos 
        acc_2 = 4.0 * np.einsum('ij,ij->i', pos, vel).reshape(-1, 1) * vel
        return fac*(acc_1 + acc_2)
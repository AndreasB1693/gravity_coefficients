import numpy as np
from numpy.linalg import norm
from numba import jit


@jit('float64[::1](float64[::1], float64)', 
     nopython=True, nogil=True, cache=True, fastmath=True)
def gravity_newton_single(pos, GM):
    """
    Calculate the acceleration due to the gravity of a point mass.

    Parameters
    ----------
    pos : numpy.ndarray, shape (3,), dtype float64
        Position vector in [km].
    GM : float
        standard gravitational parameter [km^3 s^2].

    Returns:
    -------
    acc : numpy.ndarray, shape (3,), dtype float64
        Acceleration vector  in [km/s^2]
    
    Note
    ----
    This function is optimized with Numba for improved performance.   
    """

    return -GM * pos / norm(pos)**3


@jit('float64(float64[::1], float64)', 
     nopython=True, nogil=True, cache=True, fastmath=True)
def potential_newton_single(pos, GM):
    """
    Calculate the gravity potential of an point mass.

    Parameters
    ----------
    pos : numpy.ndarray, shape (3,), dtype float64
        Position vector in [km].
    GM : float
        standard gravitational parameter [km^3 s^2].

    Returns:
    -------
    pot : float
        Gravity potential
    
    Note
    ----
    This function is optimized with Numba for improved performance.   
    """

    return GM / norm(pos)  


def newtonian_accleration(pos, GM):
    """
    Calculate the acceleration(s) due to the gravity of a point mass.

    Parameters
    ----------
    pos : numpy.ndarray
        Position vector(s) in [km]. If `pos` is 
        a 1-dimensional array, it represents a single position vector; 
        if it is 2-dimensional,each row represents a position vector.
    GM : float
        standard gravitational parameter [km^3 s^2].

    Returns
    -------
    acc : numpy.ndarray
        Acceleration vector(s) in [km/s^2]
        The shape of the output matches the shape of `pos`.

    Note
    ----
    This function automatically detects if the acceleration vector shall 
    be computed for one or multiple positions: 
    """

    if pos.ndim == 1:
        return gravity_newton_single(np.ascontiguousarray(pos), GM)
    elif pos.ndim == 2:
        return -GM * pos / norm(pos, axis=1, keepdims=True)**3
    else:
        raise ValueError("`pos` has the wrong shape!")


def newtonian_potential(pos, GM):
    """
    Calculate the gravity potential of a point mass.

    Parameters
    ----------
    pos : numpy.ndarray
        Position vector(s) in [km]. If `pos` is 
        a 1-dimensional array, it represents a single position vector; 
        if it is 2-dimensional,each row represents a position vector.
    GM : float
        standard gravitational parameter [km^3 s^2].

    Returns:
    -------
    pot : float or numpy.ndarray
        Gravity potential(s) 
        The shape of the output matches the shape of `pos`.
    
    Note
    ----
    This function automatically detects if the acceleration vector shall 
    be computed for one or multiple positions: 
    """

    if pos.ndim == 1:
        return potential_newton_single(np.ascontiguousarray(pos), GM)
    elif pos.ndim == 2:
        return GM / norm(pos, axis=1)
    else:
        raise ValueError("`pos` has the wrong shape!")


import numpy as np
from numpy.linalg import norm
from numba import jit


@jit('float64[::1](float64[::1], float64[::1],'
     'float64, float64, float64, float64)', 
     nopython=True, nogil=True, cache=True, fastmath=True)
def propergation(pos, pos_sun, nu, area, mass, c_R):
    """
    Calculate the acceleration on an satellite due to the solar 
    radiation pressure.

    Parameters
    ----------
    pos : numpy.ndarray
        Position vector of the object in the inertial frame.
    pos_sun : numpy.ndarray
        Position vector of the Sun in the inertial frame.
    nu : float
        Fractional illumination of the satellite (dimensionless).
    area : float
        Cross-sectional area of the object exposed to solar radiation
        in square meters.
    mass : float
        Mass of the object in kilograms.
    c_R : float
        Radiation pressure coefficient (dimensionless).

    Returns
    -------
    numpy.ndarray
        Solar radiation pressure acceleration vector in km/s^2.

    Note
    ----
    This function is specifically designed for use in the integration 
    process and is optimized with Numba for improved performance.
    """
    
    LIGHTSPEED = 299792458  
    SOLARLUMINOSITY = 3.828e26
    
    pos_wrt_sun = (pos - pos_sun)*1e3
    r_wrt_sun = norm(pos_wrt_sun)
    
    p_rad = SOLARLUMINOSITY / (4*np.pi*LIGHTSPEED*r_wrt_sun**2) 
    acc = - nu * p_rad * ((c_R*area) / mass) * (pos_wrt_sun / r_wrt_sun)
    
    return  acc*1e-3


def radiation_pressure(pos, pos_sun, nu, satellite):
    """
    Calculate the acceleration(s) due to the solar radiation pressure.
    
    Parameters
    ----------
    pos : numpy.ndarray
        Position vector(s) in the inertial frame. If `pos` is 
        a 1-dimensional array, it represents a single position vector; 
        if it is 2-dimensional, each row represents a position vector.
    pos_sun : numpy.ndarray
        Position vector(s) of the sun in the inertial 
        frame. If `pos_sun` is a 1-dimensional array, it represents a 
        single position vector; if it is 2-dimensional, each row 
        represents a position vector.
    nu : float or numpy.ndarray
        shadow function. IF `nu` is an array the shape must match with 
        `pos` or `pos_sun`
    sat : Satellite
        An object representing the satellite with the attributes area, 
        mass, pressure coefficient (c_R), etc.

    Returns
    -------
    acc : numpy.ndarray, shape (3,), dtype float64
        Acceleration vector(s) in [km/s^2]
        
    Note
    ----
    Calculate the acceleration(s) due to the solar radiation pressure.
    The function automatically detects if the acceleration vector shall 
    be computed for one or multiple positions:
    - pos=1d and pos_sun=1d: point is fixed, sun body is fixed.
    - pos=2d and pos_sun=1d: point is moving, sun body is fixed.
    - pos=1d and pos_sun=2d: point is fixed, sun body is moving.
    - pos=2d and pos_sun=2d: point is moving, sun body is moving.
    """

    area = satellite.area
    mass = satellite.mass
    c_R = satellite.c_R
    
    LIGHTSPEED = 299792458  
    SOLARLUMINOSITY = 3.828e26
    
    DIM_pos = pos.ndim 
    DIM_sun = pos_sun.ndim
    
    if DIM_pos == 1 and DIM_sun == 1:
        return propergation(np.ascontiguousarray(pos),
                            np.ascontiguousarray(pos_sun),
                            nu, area, mass, c_R)

    elif DIM_pos == 2 and DIM_sun == 1:
        pos_wrt_sun = (pos - pos_sun)*1e3
        r_wrt_sun = norm(pos_wrt_sun, axis=1, keepdims=True)
        p_rad = SOLARLUMINOSITY / (4*np.pi*LIGHTSPEED*r_wrt_sun**2) 
        p_rad = np.multiply(nu.reshape(-1, 1), p_rad)
        acc = - p_rad * ((c_R*area) / mass) * (pos_wrt_sun / r_wrt_sun)
        return 1e-3 * acc
       
    elif DIM_pos == 1 and DIM_sun == 2:
        pos_wrt_sun = (pos - pos_sun)*1e3
        r_wrt_sun = norm(pos_wrt_sun, axis=1, keepdims=True)
        p_rad = SOLARLUMINOSITY / (4*np.pi*LIGHTSPEED*r_wrt_sun**2) 
        p_rad = np.multiply(nu.reshape(-1, 1), p_rad)
        acc = - p_rad * ((c_R*area) / mass) * (pos_wrt_sun / r_wrt_sun)
        return 1e-3 * acc

    elif pos.shape == pos_sun.shape:
        pos_wrt_sun = (pos - pos_sun)*1e3
        r_wrt_sun = norm(pos_wrt_sun, axis=1, keepdims=True)
        p_rad = SOLARLUMINOSITY / (4*np.pi*LIGHTSPEED*r_wrt_sun**2) 
        p_rad = np.multiply(nu.reshape(-1, 1), p_rad)
        acc = - p_rad * ((c_R*area) / mass) * (pos_wrt_sun / r_wrt_sun)
        return 1e-3 * acc
    
    else:
        raise ValueError("pos or pos_3rd have the wrong shape!")

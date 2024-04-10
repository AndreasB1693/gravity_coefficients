import numpy as np
from numpy.linalg import norm
from numba import jit

@jit('float64[::1](float64[::1], float64, float64, float64, float64)',
     nopython=True, nogil=True, cache=True, fastmath=True)
def drag_single(state, rho, area, mass, c_D):
    """
    Compute the acceleration on the satellite due to atmospheric drag in
    a body-fixed reference system.

    Parameters
    ----------
    state : numpy.ndarray, shape (6,), dtype float64
        State vector of the satellite, including position
        and velocity [km, km/s] in the body-fixed frame.
    rho : float
        Atmospheric density [kg/m^3].
    area : float
        Cross-sectional area of the satellite exposed to atmospheric 
        drag in square meters.
    mass : float
        Mass of the satellite in kilograms.
    c_D : float
        Drag coefficient of the satellite.

    Returns
    -------
    numpy.ndarray, shape (3,), dtype float64
        Acceleration vector in body-fixed frame [km/s^2].


    Note
    ----
    Compute the acceleration on the satellite due to atmospheric drag 
    in a body-fixed reference system. For the computation a simple 
    "Cannonball-model" for the drag is assumed. The function is 
    optimized with Numba for improved performance. 
    """
    
    OMEGA = np.array([0.0, 0.0, 7.29212e-5])    # Earth Rotation
    BC = mass / (c_D*area)                      # Ballistic coefficient
    
    pos = state[:3]
    vel = state[3:]
    
    vel_rel = (vel - np.cross(OMEGA, pos))*1e3                        
    vel_abs = norm(vel_rel)
    acc = -(0.5 / BC) * rho * vel_abs * vel_rel
    
    return acc * 1e-03 


def atmospheric_drag(state, rho, satellite):
    """
    Compute the acceleration(s) on the satellite due to atmospheric 
    drag in a body-fixed reference system.

    Parameters
    ----------
    state : numpy.ndarray, shape (6,), dtype float64
        State vector(s) of the satellite, including position
        and velocity [km, km/s] in the body-fixed frame. If `state` is 
        a 1-dimensional array, it represents a single state vector; 
        if it is 2-dimensional, each row represents a state vector.
    rho : float or numpy.ndarray
        Atmospheric density [kg/m^3]. IF `rho` is an array the shape must match with 
        `pos` 
    sat : Satellite
        An object representing the satellite with the attributes area, 
        mass, pressure coefficient (c_R), etc.

    Returns
    -------
    numpy.ndarray, shape (3,), dtype float64
        Acceleration vector in body-fixed frame [km/s^2].
        The shape of the output matches the shape of `state`.


    Note
    ----
    Compute the acceleration on the satellite due to atmospheric drag 
    in a body-fixed reference system. For the computation a simple 
    "Cannonball-model" for the drag is assumed. The function is 
    optimized with Numba for improved performance. The function 
    automatically detects if the acceleration vector shall be computed 
    for one or multiple states
    """
    
    if state.ndim == 1:
        return drag_single(np.ascontiguousarray(state),
                           rho,
                           satellite.area,
                           satellite.mass,
                           satellite.c_D)

    elif state.ndim == 2:
        area = satellite.area
        mass = satellite.mass
        c_D = satellite.c_D
        OMEGA = np.array([0.0, 0.0, 7.29212e-5])    # Earth Rotation
        BC = mass / (c_D*area)                      # Ballistic coefficient
        
        pos = state[:, :3]
        vel = state[:, 3:]
        
        vel_rel = (vel - np.cross(OMEGA, pos)) * 1e3                        
        vel_abs = norm(vel_rel, axis=1)
        acc = -(0.5 / BC) * np.einsum('i,ij->ij', rho*vel_abs, vel_rel)
        
        return acc * 1e-03 
    else:
        raise ValueError("state has the wrong shape!")
    """
    Calculate the acceleration on the satellite due to atmospheric drag 

    Parameters
    ----------
    state : numpy.ndarray, shape (6,), dtype float64
        State vector of the satellite, including position
        and velocity [km, km/s] in the inertial frame.
    rho : float
        Atmospheric density [kg/m^3].
    area : float
        Cross-sectional area of the satellite exposed to atmospheric 
        drag in square meters.
    mass : float
        Mass of the satellite in kilograms.
    c_D : float
        Drag coefficient of the satellite.

    Returns
    -------
    numpy.ndarray, shape (3,), dtype float64
        Acceleration vector in inertial frame [km/s^2].


    Note
    ----
    This function is specifically designed for use in the integration 
    process and is optimized with Numba for improved performance.
    """
    
    if state.ndim == 1:
        return drag_single(np.ascontiguousarray(state),
                           rho,
                           satellite.area,
                           satellite.mass,
                           satellite.c_D)

    elif state.ndim == 2:
        area = satellite.area
        mass = satellite.mass
        c_D = satellite.c_D
        OMEGA = np.array([0.0, 0.0, 7.29212e-5])    # Earth Rotation
        BC = mass / (c_D*area)                      # Ballistic coefficient
        
        pos = state[:, :3]
        vel = state[:, 3:]
        
        vel_rel = (vel - np.cross(OMEGA, pos)) * 1e3                        
        vel_abs = norm(vel_rel, axis=1)
        acc = -(0.5 / BC) * np.einsum('i,ij->ij', rho*vel_abs, vel_rel)
        
        return acc * 1e-03 
    else:
        raise ValueError("state has the wrong shape!")
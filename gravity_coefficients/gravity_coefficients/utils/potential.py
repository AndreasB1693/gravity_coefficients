import numpy as np
from numpy.linalg import norm
from numba import jit, prange

@jit('float64(float64[::1], float64, float64,'
     'float64[:, :, ::1], int64, int64)', 
      nopython=True, nogil=True, cache=True, fastmath=True)
def __potential_v1(pos, ref_GM, ref_r, coeff, n_max, m_max):
    """
    Compute the harmonic potential using the Cunningham method 
    for one postion

    Parameters:
    ----------
    pos : numpy.ndarray, shape (3,), dtype float64
        Position vector in the body-fixed frame.
    ref_GM : float
        Geocentric gravitational constant (GM).
    ref_r : float
        Reference radius.
    coeff : numpy.ndarray, shape (4, n_max+1, n_max+1), dtype float64
        Coefficients for the acceleration field.
    n_max : int
        Maximum degree of spherical harmonics.
    m_max : int
        Maximum order of spherical harmonics.

    Returns:
    -------
    acc : numpy.ndarray, shape (3,), dtype float64
        Acceleration vector in km/s^2.

    Note
    ----
    This function is specifically designed for use in the integration 
    process and is optimized with Numba for improved performance.
    """

    CS = coeff[0]             # Multipole coefficients
    preVW = coeff[1]          # pre-factors of elementary potentials 

    V = np.zeros((n_max+2, n_max+2), dtype=np.float64)  
    W = np.zeros((n_max+2, n_max+2), dtype=np.float64)  
    
    # Auxiliary quantities
    radius = norm(pos)
    alpha_R = ref_r**2 / radius**2
    alpha_x = ref_r*pos[0] / radius**2  
    alpha_y = ref_r*pos[1] / radius**2  
    alpha_z = ref_r*pos[2] / radius**2
    
    # Calculate zonal terms V(n,0); set W(n,0)=0.0
    m = 0
    V[0, 0] = ref_r / radius
    V[1, 0] = preVW[1, 0]*alpha_z*V[0, 0]

    for n in prange(2, n_max + 2):
        V[n, m] = preVW[n, m]*alpha_z*V[n-1, m] - preVW[m, n]*alpha_R*V[n-2, m]

    for m in prange(1, m_max + 2):
        V[m, m] = preVW[m, m] * (alpha_x*V[m-1, m-1] - alpha_y*W[m-1, m-1])
        W[m, m] = preVW[m, m] * (alpha_x*W[m-1, m-1] + alpha_y*V[m-1, m-1])
        
        for n in prange(m+1, n_max+2):
            V[n,m] = preVW[n, m]*alpha_z*V[n-1, m] - preVW[m, n]*alpha_R*V[n-2, m]           
            W[n,m] = preVW[n, m]*alpha_z*W[n-1, m] - preVW[m, n]*alpha_R*W[n-2, m]

    # Calculate potential
    pot = 0.0
    m = 0
    for n in prange(n_max+1):
        C = CS[n, 0]  
        pot += C*V[n, m]

    for m in prange(1, m_max+1):
        for n in prange(m, n_max+1):
            C = CS[n, m]  
            S = CS[m-1, n]  
            pot += C*V[n, m] + S*W[n, m]
    
    pot *= (ref_GM/ref_r)
    return pot

@jit('float64[::1](float64[:, ::1], float64, float64,'
     'float64[:, :, ::1], int64, int64)',  
     nopython=True, nogil=True, cache=True, fastmath=True)
def __potential_v2(pos, ref_GM, ref_r, coeff, n_max, m_max):
    """
    Compute the harmonic potential using the Cunningham method 
    for mult. postion 

    Parameters:
    ----------
    pos : numpy.ndarray, shape (N, 3), dtype float64
        Position vectors in the body-fixed frame.
    ref_GM : float
        Geocentric gravitational constant (GM).
    ref_r : float
        Reference radius.
    coeff : numpy.ndarray, shape (4, n_max+1, n_max+1), dtype float64
        Coefficients for the acceleration field.
    n_max : int
        Maximum degree of spherical harmonics.
    m_max : int
        Maximum order of spherical harmonics.

    Returns:
    -------
    acc : numpy.ndarray, shape (3,), dtype float64
        Acceleration vector in km/s^2.
    """

    N = pos.shape[0]
    pot = np.zeros(N, dtype=np.float64)
    
    for i in prange(N):
        pot[i] = __potential_v1(pos[i], ref_GM, ref_r, coeff, 
                                        n_max, m_max)
    
    return pot


def harmonics(pos, model):
    """
    Calculate the harmonic potential(s) using the Cunningham method.

    Parameters:
    ----------
    pos : numpy.ndarray, shape (N ,3), dtype float64
        Position vector(s) in the body-fixed frame. If `pos` is 
        a 1-dimensional array, it represents a single position vector; 
        if it is 2-dimensional, each row represents a position vector.
    model : gravity model
        An object representing a gravity model with the attributes ref. 
        radius, stokes coefficients, ref. GM, etc. 

    Returns:
    -------
    acc : numpy.ndarray, shape (3,), dtype float64
        Acceleration vector in km/s^2.
        The shape of the output matches the shape of `pos`.

    Note
    ----
    This function automatically detects if the acceleration vector shall 
    be computed for one or multiple positions: 
    """
    DIM_pos = pos.ndim 

    if pos.ndim  == 1:
        pos = np.ascontiguousarray(pos)
        return __potential_v1(pos, 
                              model.ref_GM, 
                              model.ref_radius, 
                              model.coefficients, 
                              model.n_max, 
                              model.m_max)
    
    elif pos.ndim == 2:
        pos = np.ascontiguousarray(pos)
        return __potential_v2(pos, 
                              model.ref_GM, 
                              model.ref_radius, 
                              model.coefficients, 
                              model.n_max, 
                              model.m_max)


def newton(pos, GM):
    """
    Calculate the gravitational potential of an idealized system 
    of two point masses.

    Parameters
    ----------
    pos : numpy.ndarray
        Position vector(s) in the inertial frame. If `pos` is 
        a 1-dimensional array, it represents a single position vector; 
        if it is 2-dimensional,each row represents a position vector.
    GM : float
        Gravitational parameter (GM).

    Returns
    -------
    potential : numpy.ndarray
        Gravitational potential(s) in km^2/s^2. 
        The shape of the output matches the shape of `pos`.

    Note
    ----
    This function automatically detects whether the potential vector 
    is computed for a single position or multiple positions. 
    If `pos` is 1-dimensional, it calculates the potential for a
    single position.
    If `pos` is 2-dimensional, it calculates the potential for each 
    row representing a position vector.
    """

    if pos.ndim == 1:
        return GM / norm(pos)
    else:
        return (GM / norm(pos, axis=1, keepdims=True)).reshape(-1)
    
    
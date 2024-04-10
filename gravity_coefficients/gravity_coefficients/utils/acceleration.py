import numpy as np
from numpy.linalg import norm
from numba import jit, prange


@jit('float64[::1](float64[::1], float64, float64,'
     'float64[:, :, ::1], int64, int64)', 
      nopython=True, nogil=True, cache=True, fastmath=True)
def gravity_harmonics_single(pos, GM_REF, R_REF, coef, n_max, m_max):
    """
    Compute the acceleration due to a irregular shaped body assuming
    a harmonic potential using the Cunningham method in a body-fixed
    reference system for one position

    Parameters:
    ----------
    pos : numpy.ndarray, shape (3,), dtype float64
        Position vector in a body-fixed system in [km].
    GM_REF : float
        Reference standard gravitational parameter [km^3 s^2].
    R_REF : float
        Reference radius [km].
    coef : numpy.ndarray, shape (n_max+2, n_max+2), dtype float64
        coefficients tensor which include the stokes coefficients and 
        the compensation coefficients which decide if the gravity model 
        is normalized or unnormalized
    n_max : int
        Maximum degree of spherical harmonics.
    m_max : int
        Maximum order of spherical harmonics.

    Returns:
    -------
    acc : numpy.ndarray, shape (3,), dtype float64
        Acceleration vector in a body-fixed system in [km/s^2]

    Note
    ----
    This function is optimized with Numba for improved performance.
    """

    CS = coef[0]            # stokes coefficients
    VW = coef[1]            # compensation coefficients potentials 
    AC_XY = coef[2]         # compensation coefficients acceleration xy-plane
    AC_Z = coef[3]          # compensation coefficients acceleration z-plane


    V = np.zeros((n_max+2, n_max+2), dtype=np.float64)  
    W = np.zeros((n_max+2, n_max+2), dtype=np.float64)  
    
    # Auxiliary quantities#
    #######################
    radius = norm(pos)
    alpha_R = R_REF**2 / radius**2
    alpha_x = R_REF*pos[0] / radius**2  
    alpha_y = R_REF*pos[1] / radius**2  
    alpha_z = R_REF*pos[2] / radius**2
    
    # recurrence coefficients V and W #
    ###################################
    V[0, 0], V[1, 0] = R_REF / radius, VW[1, 0]*alpha_z*V[0, 0]

    # zonal terms
    for n in prange(2, n_max + 2):
        V[n, 0] = VW[n, 0]*alpha_z*V[n-1, 0] - VW[0, n]*alpha_R*V[n-2, 0]

    for m in prange(1, m_max + 2):
        # tesseral terms
        V[m, m] = VW[m, m] * (alpha_x*V[m-1, m-1] - alpha_y*W[m-1, m-1])
        W[m, m] = VW[m, m] * (alpha_x*W[m-1, m-1] + alpha_y*V[m-1, m-1])
        # zonal terms
        for n in prange(m+1, n_max+2):
            V[n,m] = VW[n, m]*alpha_z*V[n-1, m] - VW[m, n]*alpha_R*V[n-2, m]           
            W[n,m] = VW[n, m]*alpha_z*W[n-1, m] - VW[m, n]*alpha_R*W[n-2, m]

    # partial accelerations a_x, a_y, a_z #
    #######################################
    acc_x, acc_y, acc_z = 0.0, 0.0, 0.0

    for n in prange(n_max+1):
        C = CS[n, 0]  
        acc_x -= AC_XY[n, 0] * C * V[n+1, 1]
        acc_y -= AC_XY[n, 0] * C * W[n+1, 1]
        acc_z -= AC_Z[n, 0] * C * V[n+1, 0]

    for m in prange(1, m_max+1):
        for n in prange(m, n_max+1):

            C = CS[n, m]  
            S = CS[m-1, n]  
            
            acc_x -= (AC_XY[n, m]*(C*V[n+1, m+1] + S*W[n+1, m+1]) 
                      - AC_XY[m-1, n]*(C*V[n+1, m-1] + S*W[n+1, m-1]))
            
            acc_y -= (AC_XY[n, m] * (C*W[n+1, m+1] - S*V[n+1, m+1]) 
                      + AC_XY[m-1, n] * (C*W[n+1, m-1] - S*V[n+1, m-1]))     
            
            acc_z -= AC_Z[n, m] * (C*V[n+1, m] + S*W[n+1, m])
    
    acc = (GM_REF/(R_REF**2)) * np.array([acc_x, acc_y, acc_z])
    return acc


@jit('float64[:, ::1](float64[:, ::1], float64, float64,'
     'float64[:, :, ::1], int64, int64)',  
     nopython=True, nogil=True, cache=True, fastmath=True)
def gravity_harmonics_multi(pos, GM_REF, R_REF, coef, n_max, m_max):
    """
    Compute the acceleration due to a irregular shaped body assuming
    a harmonic potential using the Cunningham method in a body-fixed 
    reference system for multible position

    Parameters:
    ----------
    pos : numpy.ndarray
        Position vectors in a body-fixed system in [km]. 
    GM_REF : float
        Reference standard gravitational parameter [km^3 s^2].
    R_REF : float
        Reference radius [km].
    coef : numpy.ndarray, shape (n_max+2, n_max+2), dtype float64
        coefficients tensor which include the stokes coefficients and 
        the compensation coefficients which decide if the gravity model 
        is normalized or unnormalized
    n_max : int
        Maximum degree of spherical harmonics.
    m_max : int
        Maximum order of spherical harmonics.

    Returns
    -------
    acc : numpy.ndarray
        Acceleration vectors in a body-fixed system in [km/s^2]


    Note
    ----
    This function is optimized with Numba for improved performance.
    """

    N = pos.shape[0]
    acc = np.zeros((N, 3), dtype=np.float64)
    
    for i in prange(N):
        acc[i] = gravity_harmonics_single(pos[i], GM_REF, R_REF, 
                                          coef, n_max, m_max)
    
    return acc


def harmonics(pos, model):
    """
    Compute the acceleration(s) due to a irregular shaped body assuming
    a harmonic potential using the Cunningham method in a body-fixed 
    reference system for one or multible position

    Parameters:
    ----------
    pos : numpy.ndarray
        Position vector(s) in a body-fixed system in [km]. If `pos` is 
        a 1-dimensional array, it represents a single position vector; 
        if it is 2-dimensional,each row represents a position vector.
    model : gravity model
        An object representing a gravity model 

    Returns
    -------
    acc : numpy.ndarray
        Acceleration vector(s) in a body-fixed system in [km/s^2]
        The shape of the output matches the shape of `pos`.

    Note
    ----
    This function automatically detects if the acceleration vector shall 
    be computed for one or multiple positions: 
    """
    

    if pos.ndim  == 1:
        pos = np.ascontiguousarray(pos)
        return gravity_harmonics_single(pos, 
                                        model.ref_GM, 
                                        model.ref_radius, 
                                        model.coefficients, 
                                        model.n_max, 
                                        model.m_max)
    
    elif pos.ndim == 2:
        pos = np.ascontiguousarray(pos)
        return gravity_harmonics_single(pos, 
                                        model.ref_GM, 
                                        model.ref_radius, 
                                        model.coefficients, 
                                        model.n_max, 
                                        model.m_max)
        
        
def newton(pos, GM):
    """
    Calculate the gravitational acceleration for an idealized system 
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
    acc : numpy.ndarray
        Acceleration vector(s) in km/s^2. 
        The shape of the output matches the shape of `pos`.

    Note
    ----
    This function automatically detects whether the acceleration vector 
    is computed for a single position or multiple positions. 
    """

    if pos.ndim == 1:
        return -GM * pos / norm(pos)**3
    else:
        return -GM * pos / norm(pos, axis=1, keepdims=True)**3
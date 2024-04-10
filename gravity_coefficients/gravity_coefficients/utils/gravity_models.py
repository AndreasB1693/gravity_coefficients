import numpy as np
import os
import yaml
import math
import pandas as pd
from numba import jit, prange

@jit('float64[::1](float64[::1], float64, float64)', 
      nopython=True, nogil=True, cache=True, fastmath=True)
def __geodetic_v1(pos, R_EQU, FLAT):
    """
    Computes the geodetic coordinates in body-fixed coordinates system
    for one pos.
    
    Parameters
    ----------
    pos : array-like
        The position in a body-fixed coord. system
    R_EQU : float
        The equatorial radius of the reference ellipsoid.
    FLATT : float
        The flatting coef. of the body

    Returns
    -------
    geo : array-like
        The geodatic coord.
    """
    x, y, z = pos 
    rho2 = x**2 + y**2  
    
    # Iteration
    ECC2 = 2*FLAT - FLAT**2
    dz = ECC2 * z
    MAX_ITERATIONS = 10
    for _ in range(MAX_ITERATIONS):
        z_dz = z + dz
        Nh = np.sqrt(rho2 + z_dz**2)
        SinPhi = z_dz / Nh  
        N = R_EQU / np.sqrt(1.0 - ECC2 * SinPhi**2)
        dz_new = N * ECC2 * SinPhi
        dz = dz_new

    lon = np.arctan2(y, x)
    lat = np.arctan2(z_dz, np.sqrt(rho2) )
    alt = Nh - N
    return np.array([lon, lat, alt])

@jit('float64[:, ::1](float64[:, ::1], float64, float64)', 
      nopython=True, nogil=True, cache=True, fastmath=True)
def __geodetic_v2(pos, R_EQU, FLAT):
    """
    Computes the geodetic coordinates in body-fixed coordinates system
    for one pos.
    
    Parameters
    ----------
    pos : array-like
        The position in a body-fixed coord. system
    R_EQU : float
        The equatorial radius of the reference ellipsoid.
    FLATT : float
        The flatting coef. of the body

    Returns
    -------
    geo : array-like
        The geodatic coord.
    """
    N = pos.shape[0]
    geo = np.zeros((N, 3), dtype=np.float64)
    
    for i in prange(N):
        geo[i] = __geodetic_v1(pos[i], R_EQU, FLAT)
    
    return geo


def geodetic_coord(pos, R_EQU=6378.1366, FLAT=0.0033528131084554717):
    """
    Computes the geodetic coordinate(s) in a body-fixed 
    coordinates system.
    
    Parameters
    ----------
    pos : numpy.ndarray, shape (N ,3), dtype float64
        Position vector(s) in the body-fixed frame. If `pos` is 
        a 1-dimensional array, it represents a single position vector; 
        if it is 2-dimensional, each row represents a position vector.
    R_EQU : float
        The equatorial radius of the reference ellipsoid.
    FLATT : float
        The flatting coefficient of the body

    Returns
    -------
    geo : numpy.ndarray, shape (3,), dtype float64
        The geodatic coordinates
        The shape of the output matches the shape of `pos`.
    """
    if pos.ndim  == 1:
        return __geodetic_v1(np.ascontiguousarray(pos), R_EQU, FLAT)
    
    elif pos.ndim == 2:
        return __geodetic_v2(np.ascontiguousarray(pos), R_EQU, FLAT)


def add_new_gravity_model(name, body, ref_GM, ref_radius, degree, file, normalize=True):
    """
    Add a new gravity model to the dataset.

    Parameters
    ----------
    name : str
        The name of the gravity model.
    body : str
        The celestial body to which the model applies.
    ref_GM : float
        The gravitational constant of the body.
    ref_radius : float
        The reference radius.
    degree : int
        The maximum degree of the gravity model.
    file : str
        The path to the file containing gravity model data.

    Returns
    -------
    None
    """
    directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(directory, '../models/')
    path_params = os.path.join(path, name + '.yaml')
    path_u = os.path.join(path, name + '_u.npy')
    path_n = os.path.join(path, name + '_n.npy')

    parameter =  dict(
        name = name,
        body = body.lower(),
        ref_GM = ref_GM,
        ref_radius = ref_radius,
        degree = degree
        )
    with open(path_params, 'w') as outfile:
        yaml.dump(parameter, outfile, default_flow_style=False)
    outfile.close()

    if normalize == True:
        header = ['key', 'n', 'm', 'C_n', 'S_n', 'siref_GMa C', 'siref_GMa S']
        df = pd.read_csv(file, sep='\s+', names=header)
        df = df[['n', 'm', 'C_n', 'S_n']]
        df.loc[:, 'C_u'] = df.apply(lambda x: _unnormalize(x['n'],x['m']) * x['C_n'], axis=1)
        df.loc[:, 'S_u'] = df.apply(lambda x: _unnormalize(x['n'],x['m']) * x['S_n'], axis=1)
        df = df[['n', 'm', 'C_u', 'S_u', 'C_n', 'S_n']]
    else:
        header = ['key', 'n', 'm', 'C_u', 'S_u', 'siref_GMa C', 'siref_GMa S']
        df = pd.read_csv(file, sep='\s+', names=header)
        df = df[['n', 'm', 'C_u', 'S_u']]
        df.loc[:, 'C_n'] = df.apply(lambda x: _normalize(x['n'],x['m']) * x['C_u'], axis=1)
        df.loc[:, 'S_n'] = df.apply(lambda x: _normalize(x['n'],x['m']) * x['S_u'], axis=1)
        df = df[['n', 'm', 'C_u', 'S_u', 'C_n', 'S_n']]

    CS_u, CS_n = _transform(df)
    coeff_u = _get_unnoralized_coeff(degree, degree, CS_u) 
    coeff_n = _get_noralized_coeff(degree, degree, CS_n) 
    
    
    np.save(path_u, coeff_u)
    np.save(path_n, coeff_n)

def _unnormalize(n, m):
    """
    Unnormalize a pair of spherical harmonic degree and order.

    Parameters
    ----------
    n : int
        Spherical harmonic degree.
    m : int
        Spherical harmonic order.

    Returns
    -------
    float
        The unnormalized value.
    """
    n, m = int(n), int(m)
    delta_0m = 1 if m == 0 else 0
    numerator = (2 - delta_0m) * (2*n + 1) * math.factorial(n-m)
    denominator = math.factorial(n+m)
    return np.sqrt(numerator/denominator)

def _normalize(n, m):
    """
    Normalize a pair of spherical harmonic degree and order.

    Parameters
    ----------
    n : int
        Spherical harmonic degree.
    m : int
        Spherical harmonic order.

    Returns
    -------
    float
        The normalized value.
    """
    n, m = int(n), int(m)
    delta_0m = 1 if m == 0 else 0
    denominator = (2 - delta_0m) * (2*n + 1) * math.factorial(n-m)
    numerator = math.factorial(n+m)
    return np.sqrt(numerator/denominator)

def _transform(df):
    """
    Transform gravity model data into arrays.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing gravity model data.

    Returns
    -------
    tuple
        Two numpy arrays (CS_u, CS_n) representing the transformed data.
    """
    n_max = df['n'].max()
    m_max = df['m'].max()
    CS_u = np.zeros((n_max+1,m_max+1))
    CS_n = np.zeros((n_max+1,m_max+1))
    
    
    for i in range(df.shape[0]):
        n = df['n'].values[i]
        m = df['m'].values[i]
        C_u = df['C_u'].values[i]
        S_u = df['S_u'].values[i]
        C_n = df['C_n'].values[i]
        S_n = df['S_n'].values[i]
        
        if m == 0:
            CS_u[n,m] = C_u
            CS_n[n,m] = C_n
        else:
            CS_u[n,m] = C_u
            CS_u[m-1,n] = S_u
            CS_n[n,m] = C_n
            CS_n[m-1,n] = S_n
            
    return CS_u, CS_n

def _get_unnoralized_coeff(n_max, m_max, CS):
    """
    Get unnormalized coefficients for a gravity model.

    Parameters
    ----------
    n_max : int
        Maximum spherical harmonic degree.
    m_max : int
        Maximum spherical harmonic order.
    CS : np.ndarray
        Array containing transformed gravity model data.

    Returns
    -------
    np.ndarray
        Array of unnormalized coefficients.
    """
    coefficients = np.zeros((4, n_max+2, n_max+2), dtype=np.float64)
    coefficients[0][:n_max+1, :n_max+1] = CS


    m = 0
    coefficients[1, m, m] = 1
    for n in range(1, n_max + 2):
        coefficients[1, n, m] = (2*n - 1) / (n - m)
        coefficients[1, m, n] = (n + m - 1) / (n - m)
        
    for m in range(1, m_max + 2):
        coefficients[1, m,m] = (2*m-1)
        for n in range(m+1, n_max+2):
            coefficients[1, n, m] = (2*n - 1) / (n - m)
            coefficients[1, m, n] = (n + m - 1) / (n - m)
    
    m = 0
    for n in range(n_max + 1):
        coefficients[2, n, m] = 1
        coefficients[3, n, m] = (n - m + 1)
    
    for m in range(1, m_max + 1):
        for n in range(m, n_max + 1):
            coefficients[2, n, m] = 0.5 
            coefficients[2, m-1, n] = 0.5 * (n - m + 2) * (n - m + 1) 
            coefficients[3, n, m] = (n - m + 1)

    return coefficients

def _get_noralized_coeff(n_max, m_max, CS):
    """
    Get normalized coefficients for a gravity model.

    Parameters
    ----------
    n_max : int
        Maximum spherical harmonic degree.
    m_max : int
        Maximum spherical harmonic order.
    CS : np.ndarray
        Array containing transformed gravity model data.

    Returns
    -------
    np.ndarray
        Array of normalized coefficients.
    """
    coefficients = np.zeros((4, n_max+2, n_max+2), dtype=np.float64)
    coefficients[0][:n_max+1, :n_max+1] = CS


    m = 0
    coefficients[1, m, m] = 1
    for n in range(1, n_max + 2):
        fac = np.sqrt((2*n+1) / ((n+m)*(n-m)))
        coefficients[1, n, m] = fac * np.sqrt(2*n - 1)
        coefficients[1, m, n] = fac * np.sqrt(((n+m-1)*(n-m-1)) / (2*n-3))

    m = 1    
    coefficients[1, m, m] = np.sqrt(3.0)
    for n in range(2, n_max + 2):
        fac = np.sqrt((2*n+1) / ((n+m)*(n-m)))
        coefficients[1, n, m]  = fac * np.sqrt(2*n - 1)
        coefficients[1, m, n] = fac * np.sqrt(((n+m-1)*(n-m-1)) / (2*n-3))

    for m in range(2, m_max + 2):
        coefficients[1, m, m] = np.sqrt((2*m+1)/(2*m))
        for n in range(m+1, n_max+2):
            fac = np.sqrt((2*n+1) / ((n+m)*(n-m)))
            coefficients[1, n, m] = fac * np.sqrt(2*n - 1)
            coefficients[1, m, n] = fac * np.sqrt(((n+m-1)*(n-m-1)) / (2*n-3))

    m = 0
    for n in range(n_max + 1):
        coefficients[2, n, m] = np.sqrt(((2*n+1)*(n+1)*(n+2)) / (2*(2*n+3)))
        coefficients[2, m-1, n] = 0.0
        coefficients[3, n,m] =  (n+1) * np.sqrt((2*n+1)/ (2*n+3))
    
    m = 1
    for n in range(1, n_max + 1):
        fac = 0.5 * np.sqrt((2*n+1)/ (2*n+3))
        coefficients[2, n, m] = fac * np.sqrt((n+m+2)*(n+m+1))
        coefficients[2, m-1, n] = fac * np.sqrt((2*(n-m+2)*(n-m+1)))
        coefficients[3, n, m] = np.sqrt(((2*n+1)*(n+m+1)*(n-m+1))/(2*n+3))
        
    
    for m in range(2, m_max + 1):
        for n in range(m, n_max+1):
            fac = 0.5 * np.sqrt((2*n+1)/ (2*n+3)) 
            coefficients[2, n, m] = fac * np.sqrt((n+m+2)*(n+m+1))
            coefficients[2, m-1, n] = fac * np.sqrt((2*(n-m+2)*(n-m+1))/2.0) 
            coefficients[3, n, m] = np.sqrt(((2*n+1)*(n+m+1)*(n-m+1))/(2*n+3))

    return coefficients

class GravityModel:
    """
    Represents a gravitational model for a celestial body.

    Parameters
    ----------
    name : str
        The name of the gravitational model.

    Raises
    ------
    FileNotFoundError
        If the specified model does not exist.

    Attributes
    ----------
    name : str
        The name of the gravitational model.
    body : str
        The celestial body to which the model applies.
    ref_GM : float
        The gravitational constant of the body.
    ref_radius : float
        The reference radius.
    normalize : bool
        Determines whether the coefficients are normalized or not.
    degree : int
        The maximum degree of the gravitational model.
    coefficients : np.ndarray
        The coefficients of the gravitational model.

    Methods
    -------
    normalize(mode)
        Set the normalization mode of the coefficients.
    degree(value)
        Set the degree of the gravitational model.

    """
    def __init__(self, name):
        
        # Define path and check if model exists
        directory = os.path.dirname(os.path.abspath(__file__))
        self._path = os.path.join(directory, '../models/')
        if not os.path.join(directory, '../models/'):
            raise FileNotFoundError("The model does not exist!")
        
        
        path_params = os.path.join(self._path, name + '.yaml')
        path_coeff = os.path.join(self._path, name + '_n.npy')


        # Read model parameter from yaml file
        with open(path_params, "r") as outfile:
            parameters = yaml.safe_load(outfile)
        
        # Define model parameters
        self._model_name = parameters['name']
        self._body = parameters['body']
        self._ref_GM = parameters['ref_GM']
        self._ref_radius = parameters['ref_radius']
        self._degree = parameters['degree']
        self._n_max = parameters['degree']
        self._m_max = parameters['degree']
        self._normalize = True
        self._coefficients =  np.array(np.load(path_coeff), dtype=np.float64)
        outfile.close()    
          
    @property
    def name(self):
        """
        str : The name of the gravitational model.
        """
        return self._model_name  
    
    @property
    def body(self):
        """
        str : The celestial body to which the model applies.
        """
        return self._body  
    
    @property
    def ref_GM(self):
        """
        float : The gravitational constant of the body.
        """
        return self._ref_GM

    @property
    def ref_radius(self):
        """
        float : The reference radius.
        """
        return self._ref_radius 
    
    @property
    def degree(self):
        """
        int : The maximum degree of the gravitational model.
        """
        return self._degree  
    
    @property
    def coefficients(self):
        """
        np.ndarray : The coefficients of the gravitational model.
        """
        return self._coefficients

    @property
    def normalize(self):
        """
        bool : Determines whether the coefficients are normalized or not.
        """
        return self._normalize
    
    @normalize.setter
    def normalize(self, mode):
        """
        Set the normalization mode of the coefficients.

        Parameters
        ----------
        mode : bool
            The normalization mode.

        Returns
        -------
        None

        """
        if mode == True:
            self._normalize = mode
            path_coeff = os.path.join(self._path, self.name + '_n.npy')
        else:
            self._normalize = mode
            path_coeff = os.path.join(self._path, self.name + '_u.npy')
            array = np.load(path_coeff)[:, :self._n_max+2, :self._m_max+2]
        self._coefficients = np.array(array, dtype=np.float64)
     
    @property
    def n_max(self):
        """
        TODO: write doc
        """
        return self._n_max  
    
    @property
    def m_max(self):
        """
        TODO: write doc
        """
        return self._m_max 


    def set_max(self, n_max, m_max):
        """
        TODO: write doc
        """

        if n_max > self._degree or m_max > self._degree:
            raise ValueError(f'The underlying model does not support numbers '
                             + f'larger than {self._degree}')
        
        elif m_max > n_max:
            raise ValueError(f'The parameter n_max is not allowed to be ' 
                             + f'smaller than the parameter m_max')

        else:
            if self._normalize == True:
                path_coeff = os.path.join(self._path, self.name + '_n.npy')
            else:
                path_coeff = os.path.join(self._path, self.name + '_u.npy')
            
            self._n_max = n_max
            self._m_max = m_max
            array = np.load(path_coeff)[:, :self._n_max+2, :self._m_max+2]
            self._coefficients = np.array(array, dtype=np.float64)
    








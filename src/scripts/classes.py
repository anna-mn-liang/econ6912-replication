from dataclasses import dataclass

import numpy as np
import pandas as pd

# A class used to store the input variables for solving the SVAR model
@dataclass
class VARStruct:
    p: int # number of lags
    irhor: int # impulse response horizon
    vars: pd.DataFrame # DataFrame with the model variable data
    proxies: pd.DataFrame # DataFrame with the shock proxy data

# A class used to store the outputs from the SVAR model setup and IRFs calculated
@dataclass
class ModelStruct:
    m: pd.DataFrame # proxy variable vector for a single shock
    k: int # number of proxies
    Tval: int # number of observations
    n: int # number of variables
    bet: np.ndarray # VAR coefficients from OLS 
    res: np.ndarray # vector of residuals
    det: np.ndarray # vector of deterministic components
    Sigma: np.ndarray # covariance matrix
    X: np.ndarray # explanatory variable matrix
    Gamma: np.ndarray # impact matrix
    irs: pd.DataFrame # impulse response functions

# Define struct with SVAR results for confidence bands 
@dataclass
class VARciStruct:
    irsH: pd.DataFrame
    irsL: pd.DataFrame
    irs: pd.DataFrame
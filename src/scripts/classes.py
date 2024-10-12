from dataclasses import dataclass

import numpy as np
import pandas as pd

# Define VAR struct with given variables and data
@dataclass
class VARStruct:
    p: int # number of lags
    irhor: int # impulse response horizon
    vars: pd.DataFrame 
    proxies: pd.DataFrame


# Define Model struct with SVAR results
@dataclass
class ModelStruct:
    m: pd.DataFrame
    k: int
    Tval: int
    n: int
    bet: np.ndarray  
    res: np.ndarray
    det: np.ndarray
    Sigma: np.ndarray
    X: np.ndarray
    Gamma: np.ndarray
    irs: pd.DataFrame

# Define struct with SVAR results for confidence intervals
@dataclass
class VARciStruct:
    irsH: pd.DataFrame
    irsL: pd.DataFrame
    irs: pd.DataFrame
    irsHhall: pd.DataFrame
    irsLhall: pd.DataFrame
import numpy as np 
import pandas as pd

from scripts.classes import ModelStruct


# function to create a lagged dataframe 
def lagmatrix(df, lags):
    lagged_df = pd.concat([df.shift(i) for i in lags], axis=1)
    return lagged_df

# A function to calculate the IRFs when k=1
def doIRFs(AL, Gamma, scale, VAR, n):
    DT = scale * Gamma/Gamma.iloc[0]

    # Defined in MATLAB as irs = zeros(VAR.p+VAR.irhor,VAR.n,size(DT,2)); 
    # Remove the third dimension here since DT is a column 
    irs = np.zeros((VAR.p + VAR.irhor, n))

    irs[VAR.p,:] = DT[:].T
    for tt in range(1,VAR.irhor):
        lvars = irs[VAR.p + tt-1 :tt-1:-1, :].T
        irs[VAR.p + tt, :] = np.dot(lvars.T.flatten(), AL[:, :VAR.p * n].T)
    
    IRS = pd.DataFrame(irs)
    IRS.columns = VAR.vars.columns
    return IRS.iloc[VAR.p:,:].reset_index(drop=True)


# funtion VAR = doProxySVAR(VAR)
def doProxySVAR(VAR):
    lags = range(1,VAR.p+1)

    # Create lag matrix and drop rows with nans
    X_mat = lagmatrix(VAR.vars, lags)
    X_mat = X_mat.dropna().reset_index(drop=True)

    # Create Y matrix from data to start from the pth entry
    Y_mat = VAR.vars.iloc[VAR.p:].reset_index(drop=True)

    # VAR specification cont.
    m_df = VAR.proxies.iloc[VAR.p:].reset_index(drop=True)
    T_val = len(Y_mat)
    n_val = len(Y_mat.columns)
    k_val = m_df.shape[1]
    det = np.ones((VAR.vars.shape[0],1))

    # A. Run VAR
    X_with_det = np.concatenate([X_mat, det[VAR.p:,:]], axis=1)
    # VAR.bet   = [X VAR.DET(VAR.p+1:end,:)]\Y; performs a linear regression 
    # Use the normal equation which is equivalent to the \ operator in MATLAB
    bet = np.linalg.lstsq(X_with_det, Y_mat, rcond=None)[0]
    # Calculate the residuals 
    res = Y_mat - np.dot(X_with_det, bet)
    # Estimate the covariance matrix
    sigma = np.dot(res.T, res) / (T_val - bet.shape[0])

    # B. Narrative Identification
    AL = bet[:-det.shape[1],:].T
    Gamma = (res.T @ m_df)/ T_val

    scale = -1
    irs = doIRFs(AL, Gamma, scale, VAR, n_val)

    # Assign values to model struct
    modelVAR = ModelStruct(m=m_df, k=k_val, Tval=T_val, n=n_val, bet=bet, res=res, \
                           det=det, Sigma=sigma, X=X_with_det, Gamma=Gamma, irs = irs)


    return modelVAR

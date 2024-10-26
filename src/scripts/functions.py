import numpy as np 
import pandas as pd

from scripts.classes import VARStruct, ModelStruct, VARciStruct

def lagmatrix(df, lags):
    """
    lagmatrix creates a lagged matrix of the input dataframe

    Parameters:
    df: pandas DataFrame 
        Matrix to be lagged
    lags: int
        Number of lags to be included

    Returns:
    lagged_df: pandas DataFrame 
        The lagged matrix
    """
    lagged_df = pd.concat([df.shift(i) for i in lags], axis=1)
    return lagged_df

# funtion VAR = doProxySVAR(VAR)
def doProxySVAR(VAR):
    """
    doProxySVAR solves the SVAR model using the proxy variable approach.

    Parameters:
    VAR: VARStruct
        A struct containing the inputs used to solve the SVAR model

    Returns:
    modelVAR: ModelStruct
        A struct containing the outputs from the solving the SVAR model including the impulse responses
    """

    # Create explanatory variable matrix with all the lags and drop rows with nans
    lags = range(1,VAR.p+1)
    X_mat = lagmatrix(VAR.vars, lags)
    X_mat = X_mat.dropna().reset_index(drop=True)

    # Create Y matrix from data to start from the pth entry (cut away backcasted stuff)
    Y_mat = VAR.vars.iloc[VAR.p:].reset_index(drop=True)

    # VAR specification cont.
    # Create the vector of shock proxy variables
    m_df = VAR.proxies.iloc[VAR.p:].reset_index(drop=True)
    # Number of observations
    T_val = len(Y_mat)
    # Number of variables
    n_val = len(Y_mat.columns)
    # Number of proxies
    k_val = m_df.shape[1]
    det = np.ones((VAR.vars.shape[0],1))

    # A. Run VAR
    # Add all the determinist components to the explanatory variable matrix X
    X_with_det = np.concatenate([X_mat, det[VAR.p:,:]], axis=1)

    # VAR.bet   = [X VAR.DET(VAR.p+1:end,:)]\Y; performs OLS to calculate the VAR coefficients bet
    # Use the normal equation which is equivalent to the \ operator in MATLAB
    # Eta in the paper
    bet = np.linalg.lstsq(X_with_det, Y_mat, rcond=None)[0]
    # Calculate the residuals 
    res = Y_mat - np.dot(X_with_det, bet)
    # Estimate the covariance matrix
    sigma = np.dot(res.T, res) / (T_val - bet.shape[0])

    # B. Narrative (Structural) Identification with an external instrument shock
    # Create the coefficient matrix of the lagged variables
    AL = bet[:-det.shape[1],:].T
    # Calculate the impact matrix Gamma
    Gamma = (res.T @ m_df)/ T_val
    # Scale for the IRFs
    scale = -1
    # Calculate the impulse response functions and store in irs
    irs = doIRFs(AL, Gamma, scale, VAR, n_val)

    # Assign values to model struct
    modelVAR = ModelStruct(m=m_df, k=k_val, Tval=T_val, n=n_val, bet=bet, res=res, \
                           det=det, Sigma=sigma, X=X_with_det, Gamma=Gamma, irs = irs)
    
    return modelVAR


def doIRFs(AL, Gamma, scale, VAR, n):
    """
    doIRFs calculates the impulse response functions for a given SVAR model with k=1 and one shock

    Parameters:
    AL: np.ndarray
        Coefficient matrix of the lagged variables
    Gamma: np.ndarray
        Impact matrix
    scale: float
        Scaling factor for the IRFs
    VAR: VARStruct
        A struct containing the inputs used to solve the SVAR model
    n: int
        Number of variables in the model

    Returns:
    irs: pd.DataFrame
        A DataFrame containing the impulse response functions for the given SVAR model
    """
        
    # Normalise Gamma relative to its initial value
    DT = scale * Gamma/Gamma.iloc[0]

    # Defined in MATLAB as irs = zeros(VAR.p+VAR.irhor,VAR.n,size(DT,2)); 
    # Remove the third dimension here since DT is a column vector for a single shock
    irs = np.zeros((VAR.p + VAR.irhor, n))

    # Iteratively updates the irs array of impulse responeses at each time step by combining 
    # the lagged varibles lvars and the coefficient matrix AL over the entire impulse horizon
    irs[VAR.p,:] = DT[:].T
    for tt in range(1,VAR.irhor):
        lvars = irs[VAR.p + tt-1 :tt-1:-1, :].T
        irs[VAR.p + tt, :] = np.dot(lvars.T.flatten(), AL[:, :VAR.p * n].T)
    
    # Convert the generated impulse response functions to a DataFrame
    IRS = pd.DataFrame(irs)
    IRS.columns = VAR.vars.columns

    return IRS.iloc[VAR.p:,:].reset_index(drop=True)


def doProxySVARci(VAR, modelVAR, clevel, nboot, BlockSize):
    """
    doProxySVARci calculates the confidence intervals using the Moving Block Bootstrap method
    for the impulse response functions for a given SVAR model with k=1 and one shock

    Parameters:
    VAR: VARStruct
        A struct containing the inputs used to solve the SVAR model
    modelVAR: ModelStruct   
        A struct containing the outputs from the first iteration of solving the SVAR model including the impulse responses
    clevel: float
        Confidence level for the confidence intervals
    nboot: int
        Number of bootstrap samples
    BlockSize: int
        Size of blocks in the MBB bootstrap

    Returns:
    varci: VARciStruct
        A struct containing the confidence intervals for the impulse response functions for the given SVAR model
    """

    nBlock = int(np.ceil(modelVAR.Tval/BlockSize))

    # Create blocks and centerings
    Blocks = np.zeros((BlockSize, modelVAR.n, modelVAR.Tval - BlockSize + 1))
    MBlocks = np.zeros((BlockSize, modelVAR.k, modelVAR.Tval - BlockSize + 1))

    # Store blocks of residuals and proxies
    for j in range(modelVAR.Tval - BlockSize + 1):
        Blocks[:, :, j] = modelVAR.res.iloc[j:j+BlockSize, :]
        MBlocks[:, :, j] = modelVAR.m.iloc[j:j+BlockSize, :]

    # Center the data by calculating means of blocks
    centering = np.zeros((BlockSize, modelVAR.n))
    for j in range(BlockSize):
        centering[j, :] = modelVAR.res.iloc[j:modelVAR.Tval - BlockSize + j + 1, :].mean(axis=0)

    centering = np.tile(centering, (nBlock, 1))
    centering = centering[:modelVAR.Tval, :]

    # Center the boostrapped proxy variables
    Mcentering = np.zeros((BlockSize, modelVAR.k))

    # Only include the modelVAR.k=1 case. The original code includes the k=2 case. 
    for j in range(BlockSize):
        # Take subsets of the shock proxies
        subM = np.asarray(modelVAR.m.iloc[j:modelVAR.Tval - BlockSize + j + 1, :])
        subM2 = np.asarray(VAR.proxies)

        # Filter rows where the first column is not zero
        subM_filtered = subM[subM[:, 0] != 0]
        subM2_filtered = subM2[subM2[:, 0] != 0]

        # Compute the means and assign to Mcentering
        Mcentering[j, :] = subM_filtered[:, 0].mean() - subM2_filtered[:, 0].mean()

    # Replicate and truncate Mcentering to match the size of the data
    Mcentering = np.tile(Mcentering, (nBlock, 1))
    Mcentering = Mcentering[:modelVAR.Tval, :]

    jj = 1

    IRS = np.zeros((modelVAR.irs.shape[0]*modelVAR.irs.shape[1], nboot))

    while jj < nboot + 1:
        # Draw bootstrapped residuals and proxies
        # Generate random indices for selecting blocks with replacement, 
        # and equal prob on each element in the set the indices are drawn from
        index = (np.ceil(modelVAR.Tval - BlockSize + 1) * np.random.rand(nBlock, 1)).astype(int)

        U_boot = np.zeros((nBlock * BlockSize, modelVAR.n))
        M_boot = np.zeros((nBlock * BlockSize, modelVAR.k))

        # Fill bootstrap arrays with the selected blocks
        for j in range(nBlock):
            U_boot[BlockSize * j:BlockSize * (j + 1), :] = Blocks[:, :, index[j, 0] - 1]
            M_boot[BlockSize * j:BlockSize * (j + 1), :] = MBlocks[:, :, index[j, 0] - 1]
        
        U_boot = U_boot[:modelVAR.Tval, :]
        M_boot = M_boot[:modelVAR.Tval, :]

        # Center the bootstrapped residuals and proxies
        U_boot = U_boot - centering

        # Loop through bootstrapping sample of proxies and center the non-zero values so that they have zero-mean. 
        for j in range(modelVAR.k):
            non_zero_indices = M_boot[:, j] != 0
            M_boot[non_zero_indices, j] = M_boot[non_zero_indices, j] - Mcentering[non_zero_indices, j]

        resb = U_boot.T

        # Initialize the bootstrapped variables to solve the SVAR model with
        varsb = np.zeros((VAR.p + modelVAR.Tval, modelVAR.n))
        varsb[:VAR.p, :] = VAR.vars.iloc[:VAR.p, :]

        for j in range(VAR.p, VAR.p + modelVAR.Tval):
            # Extract lagged variables and update rows of varsb with new data for inference
            lvars = varsb[j-1:(None if j==VAR.p else (j-VAR.p-1)):-1, :].T
            varsb[j, :] = (lvars.T.flatten() @ modelVAR.bet[:VAR.p * modelVAR.n, :]) \
                    + (modelVAR.det[j, :] @ modelVAR.bet[VAR.p * modelVAR.n:, :]) + resb[:, j - VAR.p]

        # Storet the bootstrapped proxies and variables in DataFrames to pass to SVAR solving function
        proxies_df = pd.DataFrame(np.vstack((VAR.proxies.iloc[:VAR.p, :], M_boot)))
        proxies_df.columns = VAR.proxies.columns
        varsb_df = pd.DataFrame(varsb)
        varsb_df.columns = VAR.vars.columns

        # Initalize SVAR struct with parameters and bootstrapped data
        VARBS = VARStruct(p=VAR.p, irhor=VAR.irhor, vars=varsb_df, proxies=proxies_df)

        # Solve the SVAR model with bootstrapped data
        modelVARBS = doProxySVAR(VARBS)

        # Store the IRFs from solving with bootstrapped data
        irs = np.asarray(modelVARBS.irs.iloc[:, :])
        IRS[:, jj-1] = irs.T.flatten()

        jj += 1


    # Initialize irsH, irsL, and irs with zeros
    irsHBS = np.zeros((VAR.irhor, modelVAR.irs.shape[1]))
    irsLBS = np.zeros((VAR.irhor, modelVAR.irs.shape[1]))
    irsBS = np.zeros((VAR.irhor, modelVAR.irs.shape[1]))

    # Calculate confidence intervals
    irsHBS[:, :] = np.quantile(IRS[:, :].T, (1 - clevel / 100) / 2, axis=0).reshape(modelVAR.irs.shape[1], VAR.irhor).T
    irsLBS[:, :] = np.quantile(IRS[:, :].T, 1 - (1 - clevel / 100) / 2, axis=0).reshape(modelVAR.irs.shape[1], VAR.irhor).T
    irsBS[:, :] = np.quantile(IRS[:, :].T, 1 - (1 - 50 / 100), axis=0).reshape(modelVAR.irs.shape[1], VAR.irhor).T

    # Store the bootstrapped IRFs in DataFrames
    irsHBS_df = pd.DataFrame(irsHBS)
    irsLBS_df = pd.DataFrame(irsLBS)
    irsBS_df = pd.DataFrame(irsBS)

    irsHBS_df.columns = modelVAR.irs.columns
    irsLBS_df.columns = modelVAR.irs.columns
    irsBS_df.columns = modelVAR.irs.columns

    VARci = VARciStruct(irsH=irsHBS_df, irsL=irsLBS_df, irs=irsBS_df)

    return VARci
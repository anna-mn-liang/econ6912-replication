import numpy as np 
import pandas as pd

from scripts.classes import VARStruct, ModelStruct, VARciStruct

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


def doProxySVARci(VAR, modelVAR, clevel, nboot, BlockSize):
    nBlock = int(np.ceil(modelVAR.Tval/BlockSize))

    # Create blocks and centerings
    Blocks = np.zeros((BlockSize, modelVAR.n, modelVAR.Tval - BlockSize + 1))
    MBlocks = np.zeros((BlockSize, modelVAR.k, modelVAR.Tval - BlockSize + 1))

    for j in range(modelVAR.Tval - BlockSize + 1):
        Blocks[:, :, j] = modelVAR.res.iloc[j:j+BlockSize, :]
        MBlocks[:, :, j] = modelVAR.m.iloc[j:j+BlockSize, :]

    # Center the bootstrapped VAR errors
    centering = np.zeros((BlockSize, modelVAR.n))
    for j in range(BlockSize):
        centering[j, :] = modelVAR.res.iloc[j:modelVAR.Tval - BlockSize + j + 1, :].mean(axis=0)

    centering = np.tile(centering, (nBlock, 1))
    centering = centering[:modelVAR.Tval, :]

    # Center the boostrapped proxy variables
    Mcentering = np.zeros((BlockSize, modelVAR.k))

    # Only include the modelVAR.k=1 case. The original code includes the k=2 case. 
    for j in range(BlockSize):
        subM = np.asarray(modelVAR.m.iloc[j:modelVAR.Tval - BlockSize + j + 1, :])
        subM2 = np.asarray(VAR.proxies)

        # Filter rows where the first column is not zero
        subM_filtered = subM[subM[:, 0] != 0]
        subM2_filtered = subM2[subM2[:, 0] != 0]

        # Compute the means and assign to Mcentering
        Mcentering[j, :] = subM_filtered[:, 0].mean() - subM2_filtered[:, 0].mean()

    Mcentering = np.tile(Mcentering, (nBlock, 1))
    Mcentering = Mcentering[:modelVAR.Tval, :]

    jj = 1

    IRS = np.zeros((modelVAR.irs.shape[0]*modelVAR.irs.shape[1], nboot))

    while jj < nboot + 1:
        # draw bootstrapped residuals and proxies
        # generate random indices
        index = (np.ceil(modelVAR.Tval - BlockSize + 1) * np.random.rand(nBlock, 1)).astype(int)

        U_boot = np.zeros((nBlock * BlockSize, modelVAR.n))
        M_boot = np.zeros((nBlock * BlockSize, modelVAR.k))

        for j in range(nBlock):
            U_boot[BlockSize * j:BlockSize * (j + 1), :] = Blocks[:, :, index[j, 0] - 1]
            M_boot[BlockSize * j:BlockSize * (j + 1), :] = MBlocks[:, :, index[j, 0] - 1]
        
        U_boot = U_boot[:modelVAR.Tval, :]
        M_boot = M_boot[:modelVAR.Tval, :]

        # Center the bootstrapped residuals and proxies
        U_boot = U_boot - centering

        for j in range(modelVAR.k):
            non_zero_indices = M_boot[:, j] != 0
            M_boot[non_zero_indices, j] = M_boot[non_zero_indices, j] - Mcentering[non_zero_indices, j]

        resb = U_boot.T

        varsb = np.zeros((VAR.p + modelVAR.Tval, modelVAR.n))
        varsb[:VAR.p, :] = VAR.vars.iloc[:VAR.p, :]

        for j in range(VAR.p, VAR.p + modelVAR.Tval):
            lvars = varsb[j-1:(None if j==VAR.p else (j-VAR.p-1)):-1, :].T
            varsb[j, :] = (lvars.T.flatten() @ modelVAR.bet[:VAR.p * modelVAR.n, :]) \
                    + (modelVAR.det[j, :] @ modelVAR.bet[VAR.p * modelVAR.n:, :]) + resb[:, j - VAR.p]

        proxies_df = pd.DataFrame(np.vstack((VAR.proxies.iloc[:VAR.p, :], M_boot)))
        proxies_df.columns = VAR.proxies.columns

        varsb_df = pd.DataFrame(varsb)
        varsb_df.columns = VAR.vars.columns


        VARBS = VARStruct(p=VAR.p, irhor=VAR.irhor, vars=varsb_df, proxies=proxies_df)

        modelVARBS = doProxySVAR(VARBS)

        # for i in range(modelVARBS.irs.shape[2]):
        irs = np.asarray(modelVARBS.irs.iloc[:, :])
        IRS[:, jj-1] = irs.T.flatten()

        jj += 1


    # Initialize irsH, irsL, and irs with zeros
    irsHBS = np.zeros((VAR.irhor, modelVAR.irs.shape[1]))
    irsLBS = np.zeros((VAR.irhor, modelVAR.irs.shape[1]))
    irsBS = np.zeros((VAR.irhor, modelVAR.irs.shape[1]))

    # Initialize irsHhall and irsLhall with zeros
    irsHhallBS = np.zeros((VAR.irhor, modelVAR.irs.shape[1]))
    irsLhallBS = np.zeros((VAR.irhor, modelVAR.irs.shape[1]))

    # Confidence Bands
    ##################
    irsHBS[:, :] = np.quantile(IRS[:, :].T, (1 - clevel / 100) / 2, axis=0).reshape(modelVAR.irs.shape[1], VAR.irhor).T
    irsLBS[:, :] = np.quantile(IRS[:, :].T, 1 - (1 - clevel / 100) / 2, axis=0).reshape(modelVAR.irs.shape[1], VAR.irhor).T
    irsBS[:, :] = np.quantile(IRS[:, :].T, 1 - (1 - 50 / 100), axis=0).reshape(modelVAR.irs.shape[1], VAR.irhor).T

    # virs = np.asarray(modelVAR.irs.iloc[:, :])
    # irsHhallBS[:, :] = modelVAR.irs.iloc[:, :] - np.quantile((IRS[ :, :].T - np.tile(virs.T.flatten(), (nboot, 1))), (1 - clevel / 100) / 2, axis=0).reshape(VAR.irhor, modelVAR.irs.shape[1])
    # irsLhallBS[:, :] = modelVAR.irs.iloc[:, :] - np.quantile((IRS[:, :].T - np.tile(virs.T.flatten(), (nboot, 1))), 1 - (1 - clevel / 100) / 2, axis=0).reshape(VAR.irhor, modelVAR.irs.shape[1])

    irsHBS_df = pd.DataFrame(irsHBS)
    irsLBS_df = pd.DataFrame(irsLBS)
    irsBS_df = pd.DataFrame(irsBS)
    irsHhallBS_df = pd.DataFrame(irsHhallBS)
    irsLhallBS_df = pd.DataFrame(irsLhallBS)

    irsHBS_df.columns = modelVAR.irs.columns
    irsLBS_df.columns = modelVAR.irs.columns
    irsBS_df.columns = modelVAR.irs.columns
    irsHhallBS_df.columns = modelVAR.irs.columns
    irsLhallBS_df.columns = modelVAR.irs.columns

    VARci = VARciStruct(irsH=irsHBS_df, irsL=irsLBS_df, irs=irsBS_df, irsHhall=irsHhallBS_df, irsLhall=irsLhallBS_df)

    return VARci
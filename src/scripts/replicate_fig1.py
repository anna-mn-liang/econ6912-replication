# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from econ_lib import data_dir
print(data_dir)

import numpy as np 
import pandas as pd

# %%
mdata_filepath = data_dir / "monthly_data.xlsx"
mdata = pd.read_excel(mdata_filepath)
mdata

# %%
# Apply transformations to full dataset
# Scale CPI, IP, rents, CPInet, PCE, PCEnet
mdata.loc[:, ["CPI", "IP"]] = np.log(mdata.loc[:,["CPI", "IP"]])*100
mdata.loc[:,["rents", "cpinet", "pce", "pcenet", "shelter"]] = np.log(mdata.loc[:,["rents", "cpinet", "pce", "pcenet", "shelter"]])*100
mdata = mdata.fillna(0)
mdata = mdata.query("year >=1983 & month>=1")
mdata

# %%
pdata_filepath = data_dir / "monthly_factors.xlsx"
pdata = pd.read_excel(pdata_filepath)
pdata

# %%
ppdata = mdata[['year', 'month']]
ppdata = ppdata.merge(pdata, on=['year', 'month'], how='left')
ppdata = ppdata.fillna(0)
ppdata

# %%
from dataclasses import dataclass
from scripts.classes import VARStruct

# Define VAR struct with hardcoded variables
VAR = VARStruct(p=12, irhor=48, vars=mdata[['gs1', 'IP', 'ebp', 'CPI', 'rents']], proxies=ppdata[['ff4_tc']])

#%%
from scripts.classes import ModelStruct
from scripts.functions import doProxySVAR

# %%
modelVAR = doProxySVAR(VAR)
modelVAR.irs

# %%
# Inference parameters
nboot = 5000; # Number of Bootstrap Samples (Paper does 5000)
clevel = 68; # Bootstrap Percentile Shown
BlockSize = int(np.floor(5.03*len(mdata)**0.25)); # size of blocks in the MBB bootstrap
seed = 2; # seed for random number generator
np.random.seed(seed)


# %%
# Function to do 6: Jentsch and Lunsford Moving Block Bootstrap (adjusted to allow non zero-mean proxies)
# function VARci = doProxySVARci(VAR,clevel,method,nboot,BlockSize)

# def doProxySVARci(VAR, modelVAR, clevel, method, nboot, BlockSize):
#     irsL = np.nan * np.zeros((modelVAR.irs.shape[0], modelVAR.irs.shape[1], clevel))
#     irsH = np.nan * np.zeros((modelVAR.irs.shape[0], modelVAR.irs.shape[1], clevel))
#     # Newey West Lags
#     NWlags = np.floor(4*(((len(VAR.vars) - VAR.p)/100) ** (2/9)))

#     VARci = VARciStruct(irsH=irsH, irsL=irsL, irs=modelVAR.irs, irsHhall=irsH, irsLhall=irsL)

    
#     return VARci

# %%
# VARci = doProxySVARci(VAR, modelVAR, clevel, 'MBB', nboot, BlockSize)

# %%
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

# %%
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

# %%
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


# %%
# Initialize irsH, irsL, and irs with zeros
irsHBS = np.zeros((VAR.irhor, modelVAR.irs.shape[1]))
irsLBS = np.zeros((VAR.irhor, modelVAR.irs.shape[1]))
irsBS = np.zeros((VAR.irhor, modelVAR.irs.shape[1]))

# Initialize irsHhall and irsLhall with zeros
irsHhallBS = np.zeros((VAR.irhor, modelVAR.irs.shape[1]))
irsLhallBS = np.zeros((VAR.irhor, modelVAR.irs.shape[1]))

# %%
# Confidence Bands
##################
irsHBS[:, :] = np.quantile(IRS[:, :].T, (1 - clevel / 100) / 2, axis=0).reshape(modelVAR.irs.shape[1], VAR.irhor).T
irsLBS[:, :] = np.quantile(IRS[:, :].T, 1 - (1 - clevel / 100) / 2, axis=0).reshape(modelVAR.irs.shape[1], VAR.irhor).T
irsBS[:, :] = np.quantile(IRS[:, :].T, 1 - (1 - 50 / 100), axis=0).reshape(modelVAR.irs.shape[1], VAR.irhor).T

# virs = np.asarray(modelVAR.irs.iloc[:, :])
# irsHhallBS[:, :] = modelVAR.irs.iloc[:, :] - np.quantile((IRS[ :, :].T - np.tile(virs.T.flatten(), (nboot, 1))), (1 - clevel / 100) / 2, axis=0).reshape(VAR.irhor, modelVAR.irs.shape[1])
# irsLhallBS[:, :] = modelVAR.irs.iloc[:, :] - np.quantile((IRS[:, :].T - np.tile(virs.T.flatten(), (nboot, 1))), 1 - (1 - clevel / 100) / 2, axis=0).reshape(VAR.irhor, modelVAR.irs.shape[1])

# %%
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

# %%
from scripts.classes import VARciStruct
VARci = VARciStruct(irsH=irsHBS_df, irsL=irsLBS_df, irs=irsBS_df, irsHhall=irsHhallBS_df, irsLhall=irsLhallBS_df)

# %%
shocksize = -0.25
shock = 1

# %%
modelVAR.irs = modelVAR.irs * shocksize
VARci.irsH = VARci.irsH * shocksize
VARci.irsL = VARci.irsL * shocksize
VARci.irs = VARci.irs * shocksize

# %%
import matplotlib.pyplot as plt

plotdisplay = ['IP', 'RENTS', 'CPI', 'GS1', 'EBP']
FigLabels = ['Industrial Production', 'Rents', 'CPI', 'One-Year Rate', 'Excess Bond Premium']

display1 = np.array([1,4,3,0,2])

fig, axes = plt.subplots(3, 2, figsize=(8,8), constrained_layout=True)
axes = axes.flatten()

for nvar in range(len(display1)):
    if VARci is not None:
        ax = axes[nvar]
        h, = ax.plot(range(VAR.irhor), VARci.irs.iloc[:, display1[nvar]], 'r', linewidth=1.5)
        
        if nvar == 1:
            ax.set_ylim([-0.2, 0.6])
            ax.set_yticks(np.arange(-0.2, 0.7, 0.2))
        
        ax.set_xlim([0, VAR.irhor - 1])
        
        if VARci.irsH is not None:
            ax.fill_between(range(VAR.irhor), VARci.irsH.iloc[:, display1[nvar]], VARci.irsL.iloc[:, display1[nvar]], color='blue', alpha=0.15)
        
        ax.axhline(0, color='k', linestyle='-')
        title_text = FigLabels[nvar]
        ax.set_title(title_text, fontsize=12)
        ax.set_xticks(np.arange(0, VAR.irhor, 6))
        ax.grid(True)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

axes.flat[-1].set_visible(False) # to remove last plot

plt.tight_layout()
plt.show()

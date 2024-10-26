# ---
# Title: Replicate Figure 1 of Dias and Duarte (2019)
# Author: Anna Liang
# Description: This python script is based off Dias and Duarte's MATLAB script Replicate_Figure1.m
#   It solves the SVAR model and uses the MBB method to re-sample for the 68% confidence bands. 
# Inputs: N/A
# Output: Shows Figure 1 and saves as png to the outputs folder

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
from econ_lib import data_dir, outputs_dir
print(data_dir)

import numpy as np 
import pandas as pd

# %%
## Read and transform monthly data
mdata_filepath = data_dir / "monthly_data.xlsx"
mdata = pd.read_excel(mdata_filepath)

# Convert CPI, IP, rents, CPInet, PCE, PCEnet to a rate
mdata.loc[:, ["CPI", "IP"]] = np.log(mdata.loc[:,["CPI", "IP"]])*100
mdata.loc[:,["rents", "cpinet", "pce", "pcenet", "shelter"]] = np.log(mdata.loc[:,["rents", "cpinet", "pce", "pcenet", "shelter"]])*100
mdata = mdata.fillna(0)
mdata = mdata.query("year >=1983 & month>=1")

# %%
## Read monthly shock identification data
pdata_filepath = data_dir / "monthly_factors.xlsx"
pdata = pd.read_excel(pdata_filepath)

# %%
## Align shock series with dates from monthly data
ppdata = mdata[['year', 'month']]
ppdata = ppdata.merge(pdata, on=['year', 'month'], how='left')
ppdata = ppdata.fillna(0)
ppdata

# %%
from scripts.classes import VARStruct
from scripts.functions import doProxySVAR, doProxySVARci

# %%
## Define model and inference parameters
p = 12; # number of lags
irhor = 48; # impulse response horizon
nboot = 5000; # Number of Bootstrap Samples (Paper does 5000)
clevel = 68; # Bootstrap Percentile Shown
BlockSize = int(np.floor(5.03*len(mdata)**0.25)); # size of blocks in the MBB bootstrap
seed = 2; # seed for random number generator
np.random.seed(seed)

# Define VAR struct to feed into model
VAR = VARStruct(p, irhor, vars=mdata[['gs1', 'IP', 'ebp', 'CPI', 'rents']], proxies=ppdata[['ff4_tc']])

# %%
## Solve the SVAR model
modelVAR = doProxySVAR(VAR)

## Resample using the Moving Block Bootstrap method 
VARci = doProxySVARci(VAR, modelVAR, clevel, nboot, BlockSize)

# %%
# Set a contractionary 25bps MP shock and scale the impulse responses
shocksize = -0.25 
shock = 1

modelVAR.irs = modelVAR.irs * shocksize
VARci.irsH = VARci.irsH * shocksize
VARci.irsL = VARci.irsL * shocksize
VARci.irs = VARci.irs * shocksize

# %%
## Plot the median IRFs and the 68% confidence bands
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
plt.savefig(outputs_dir / 'Figure1_US.png', dpi=300)
plt.show()


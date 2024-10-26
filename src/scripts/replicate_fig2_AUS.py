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
from econ_lib import data_dir, outputs_dir
print(data_dir)

import numpy as np 
import pandas as pd

# %%
## Read and transform quarterly data
mdata_filepath = data_dir / "quarterly_data_AUS.xlsx"
mdata = pd.read_excel(mdata_filepath)
mdata = mdata.query("year<=2020 & quarter<=3")

# Rename columns
colnames = ['year', 'quarter', 'CPI', 'IP', 'DFD', '2YRate', '3YSpread', 'HP', 'rents', 'HO']
mdata.columns = colnames

# Log the index variables
mdata.loc[:, ["CPI", "IP", "DFD", "HP", "rents"]] = np.log(mdata.loc[:,["CPI", "IP", "DFD", "HP", "rents"]])*100

# %%
## Read quarterly shock identification data
pdata_filepath = data_dir / "quarterly_shocks_AUS.xlsx"
pdata = pd.read_excel(pdata_filepath)

# %%
## Align shock series with dates from quarterly data
ppdata = mdata[['year', 'quarter']]
ppdata = ppdata.merge(pdata, on=['year', 'quarter'], how='left')
ppdata = ppdata.fillna(0)
ppdata


# %%
from dataclasses import dataclass
from scripts.classes import VARStruct, ModelStruct, VARciStruct
# import sys
# sys.path.append('/Users/annaliang/Documents/Repositories/econ6912-replication/src/scripts')
from scripts.functions import doProxySVAR, doProxySVARci

varlist = ['HP', 'rents', 'HO']

## Define model and inference parameters
p = 2; # number of lags
irhor = 20; # impulse response horizon
nboot = 5000; # Number of Bootstrap Samples (Paper does 5000)
clevel = 68; # Bootstrap Percentile Shown
BlockSize = int(np.floor(5.03*len(mdata)**0.25)); # size of blocks in the MBB bootstrap
seed = 2; # seed for random number generator
np.random.seed(seed)

# Initialize arrays to store IRFs from each iteration of the SVAR-solving loop
irs_plot = np.zeros((irhor, len(varlist)))
irsH_plot = np.zeros((irhor, len(varlist)))
irsL_plot = np.zeros((irhor, len(varlist)))

# Solve the SVAR with the common core variables and one added housing variable each iteration
for x_var in varlist:
    # Define VAR struct with hardcoded variables
    VAR = VARStruct(p, irhor, vars=mdata[['CPI', 'IP', '2YRate', '3YSpread', x_var]], proxies=ppdata[['agg_shock']])

    # Solve the SVAR model
    modelVAR = doProxySVAR(VAR)

    # Resample using the Moving Block Bootstrap method
    VARci = doProxySVARci(VAR, modelVAR, clevel, nboot, BlockSize)

    # Set a contractionary 25bps MP shock and scale the impulse responses
    shocksize = -0.25
    shock = 1

    modelVAR.irs = modelVAR.irs * shocksize
    VARci.irsH = VARci.irsH * shocksize
    VARci.irsL = VARci.irsL * shocksize
    VARci.irs = VARci.irs * shocksize

    # Store the IRFs in the arrays to plot
    irs_plot[:, varlist.index(x_var)] = VARci.irs[x_var]
    irsH_plot[:, varlist.index(x_var)] = VARci.irsH[x_var]
    irsL_plot[:, varlist.index(x_var)] = VARci.irsL[x_var]

irs_plot = pd.DataFrame(irs_plot, columns=varlist)
irsH_plot = pd.DataFrame(irsH_plot, columns=varlist)
irsL_plot = pd.DataFrame(irsL_plot, columns=varlist)

VARci_plot = VARciStruct(irsH=irsH_plot, irsL=irsL_plot, irs=irs_plot)

# %%
## Plot the median IRFs and the 68% confidence bands
import matplotlib.pyplot as plt

plotdisplay = ['HP', 'rents', 'HO']
FigLabels = ['Housing Prices', 'Housing Rents', 'Homeownership Rate']

display1 = np.array([0, 1, 2])
# display1 = np.array([0])

fig, axes = plt.subplots(2, 2, figsize=(10,8), constrained_layout=True)
axes = axes.flatten()

for nvar in range(len(display1)):
    if VARci is not None:
        ax = axes[nvar]
        h, = ax.plot(range(VAR.irhor), VARci_plot.irs.iloc[:, display1[nvar]], 'r', linewidth=1.5)
        
        ax.set_xlim([0, VAR.irhor - 1])
        
        if VARci_plot.irsH is not None:
            ax.fill_between(range(VAR.irhor), VARci_plot.irsH.iloc[:, display1[nvar]], VARci_plot.irsL.iloc[:, display1[nvar]], color='blue', alpha=0.15)
        
        ax.axhline(0, color='k', linestyle='-')
        title_text = FigLabels[nvar]
        ax.set_title(title_text, fontsize=14)
        ax.set_xticks(np.arange(0, VAR.irhor, 6))
        ax.set_xticklabels(ax.get_xticks(), fontsize=12)
        ax.grid(True)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)

axes.flat[-1].set_visible(False) # to remove last plot

plt.tight_layout()
plt.savefig(outputs_dir / 'Figure2_AUS.png', dpi=300)
plt.show()

# %%

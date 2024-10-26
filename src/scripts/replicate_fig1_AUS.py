# ---
# Title: Replicate Figure 1 of Dias and Duarte (2019) using Australian data
# Author: Anna Liang
# Description: This python script uses the functions from relicate_fig1.py 
#   to replicate Figure 1 of Dias and Duarte (2019) using Australian data.
#   It solves the SVAR model and uses the MBB method to re-sample for the 68% confidence bands. 
# Inputs: Data files quarterly_data_AUS.xlsx and quarterly_shocks_AUS.xlsx
# Output: Shows the impulse responses and saves as png to the outputs folder

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

# Rename columns
colnames = ['year', 'quarter', 'CPI', 'IP', 'DFD', '2YRate', '3YSpread', 'HP', 'rents', 'HO']
mdata.columns = colnames

# Log the index variables
mdata.loc[:, ["CPI", "IP", "DFD", "rents"]] = np.log(mdata.loc[:,["CPI", "IP", "DFD", "rents"]])*100

# %%
# Plot Australian data to check for trending series
import matplotlib.pyplot as plt

# Determine the number of subplots needed
num_plots = len(mdata.columns) - 2  # Exclude 'year' and 'quarter'
num_cols = 3  # Number of columns in the subplot grid
num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
axes = axes.flatten()  # Flatten the axes array for easy iteration

for i, column in enumerate(mdata.columns):
    if column not in ['year', 'quarter']:
        ax = axes[i-1]
        ax.plot(mdata['year'] + (mdata['quarter'] - 1) / 4, mdata[column], label=column)
        ax.set_xlabel('Year')
        ax.set_ylabel(column)
        ax.set_title(f'Time Series of {column}')
        ax.legend()
        ax.grid(True)

# Remove any unused subplots
axes.flat[0].set_visible(False) # to remove last plot

plt.tight_layout()
plt.savefig(outputs_dir / 'Check_AUS_data.png', dpi=300)

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
from scripts.classes import VARStruct
from scripts.functions import doProxySVAR, doProxySVARci

# %%
## Define model and inference parameters
p = 2; # number of lags
irhor = 20; # impulse response horizon
nboot = 5000; # Number of Bootstrap Samples (Paper does 5000)
clevel = 68; # Bootstrap Percentile Shown
BlockSize = int(np.floor(5.03*len(mdata)**0.25)); # size of blocks in the MBB bootstrap
seed = 2; # seed for random number generator
np.random.seed(seed)

# Define VAR struct to feed into model
VAR = VARStruct(p, irhor, vars=mdata[['CPI', 'IP', '2YRate', '3YSpread', 'rents']], proxies=ppdata[['agg_shock']])
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

plotdisplay = ['DFD', 'rents', 'CPI', '2YRate', '3YSpread']
FigLabels = ['Domestic Final Demand', 'Rents', 'CPI', 'Two-Year Rate', 'Three-Year Spread']

display1 = np.array([1,4,3,0,2])

fig, axes = plt.subplots(3, 2, figsize=(8,8), constrained_layout=True)
axes = axes.flatten()

for nvar in range(len(display1)):
    if VARci is not None:
        ax = axes[nvar]
        h, = ax.plot(range(VAR.irhor), VARci.irs.iloc[:, display1[nvar]], 'r', linewidth=1.5)
        
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
plt.savefig(outputs_dir / 'Figure1_AUS.png', dpi=300)
plt.show()


# %%

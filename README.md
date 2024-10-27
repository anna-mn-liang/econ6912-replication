# ECON6912 Replication Project
This repository provides the data and code required to replicate Figures 1 and 2 from Dias and Duarte's (2019) paper [Monetary policy, housing rents and inflation dynamics](https://onlinelibrary.wiley.com/doi/abs/10.1002/jae.2679). It also provides the data and codes to apply the model from this paper to an Australian setting.

## Installation
The code is in Python and requires the package manager uv to run. Install instructions are (here)[https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2]. 


## Usage
In the project directory, run the following commands in the terminal to generate the output figures.

```console
uv run src/scripts/replicate_fig1.py 
uv run src/scripts/replicate_fig2.py 
uv run src/scripts/replicate_fig1_AUS.py 
uv run src/scripts/replicate_fig2_AUS.py 
``` 

To run the Jupyter notebook for cleaning the AUS data, run `uv run jupyter lab` and open the file `/src/scripts/data_cleaning.ipynb`. In the notebook, 'Run All'. 



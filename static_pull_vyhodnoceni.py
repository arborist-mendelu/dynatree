#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 06:21:13 2024

Pracuje s daty z  csv_output/regresions_static.csv, vytvoreneho skriptem 
static_pull.py


@author: marik
"""

import pandas as pd
import numpy as np

df = pd.read_csv("csv_output/regresions_static.csv", index_col=0)
df["measurement"] = "M0"+ df["measurement"].str[-1]
df = df.drop(['p-value', 'stderr', 'intercept_stderr'], axis=1)
df = df[~((df["Dependent"].str.contains("Pt")) & (df["measurement"]=="M01"))]

# Ignore undefined values
mask = df.isnull().any(axis=1)
df_failed = df[mask]
df = df[np.logical_not(mask)]
df_failed


# Split _Rope and _Measure
dfs = {}
for method in ["Rope", "Measure"]:
    mask = df["Independent"].str.contains(method)
    dfs[method] = df[mask.values]
    dfs[method].loc[:,"Independent"] = dfs[method].loc[:,"Independent"].str.replace(f"_{method}","")
    col_dict = {
        "Slope": f"Slope_{method}",
        "Intercept": f"Intercept_{method}",
        "R^2": f"R^2_{method}",
        }
    dfs[method] = dfs[method].rename(columns=col_dict)

df_both = pd.merge(dfs["Rope"], dfs["Measure"], on=[
    'Independent', 'Dependent', 'pull', 'date', 'tree', 'measurement'])

#%%

df = df_both.copy()

mask = df["Dependent"].str.contains("_Min")
df_Min = df.loc[mask,:]
df = df.loc[np.logical_not(mask),:]

df.loc[:,"R^2_diff"] = np.abs(df["R^2_Rope"] - df["R^2_Measure"])
df = df.sort_values(by="R^2_diff")

ax = df["R^2_diff"].hist()
ax.set(yscale='log')

df_tail = df.tail(n=20)

 #%%
mask = df["Dependent"].str.contains("_Min")
df_Min = df.iloc[mask,:]
df = df.iloc[~mask,:]
df_Min
#%%
# Průměrný R^2 pro minor uhly a pro zbytek
df_Min["R^2"].mean(), df["R^2"].mean()

#%%

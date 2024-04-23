#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 17:37:18 2024

@author: marik
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob
import lib_dynatree as ld

fft_files = glob.glob("fft_data*.xlsx")
dfs = {}
for i in fft_files:
    day = ld.directory2date(ld.date2dirname(i.split("_")[5]))
    data = pd.read_excel(i)
    data["date"] = day
    data[["tree","measurement"]] = data.iloc[:,0].str.split("_",expand=True)
    dfs[i]=data
    
df_f = pd.concat(dfs,ignore_index=True)
df_f = df_f[["date","tree","measurement","Freq"]]


#%%

# data.ilo[0].split("_")
# data

#%%

for method in ["hilbert", "peaks"]:
    df = pd.read_csv(f"damping_{method}/damping_results.csv")
    
    dfm = df.merge(df_f)
    
    dfm["damping"] = -dfm["k"]/dfm["Freq"]
    
    fig, ax = plt.subplots(figsize=(15,6))
    
    sns.boxplot(
        data=dfm, 
        x="tree", 
        y="damping", 
        hue="date",
        ax = ax
        )
    [ax.axvline(x+.5,color='gray', lw=1.5) for x in ax.get_xticks()]
    ax.legend()
    ax.set(title=f"Utlum. Modrá a zelená bez listí, hnědá a červená s listím. Method: {method}")
    ax.grid(alpha=0.4)

#%%


# fig, ax = plt.subplots(figsize=(15,6))

# sns.swarmplot(
#     data=dfm, 
#     x="tree", 
#     y="k", 
#     hue="date",
#     ax = ax
#     )
# [ax.axvline(x+.5,color='gray', lw=1.5) for x in ax.get_xticks()]
# ax.legend()
# ax.set(title="Utlum. Modrá a zelená bez listí, hnědá a červená s listím")
# ax.grid(alpha=0.4)

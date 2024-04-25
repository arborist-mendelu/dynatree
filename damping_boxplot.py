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
import os

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
    fig.savefig(f"bara/{method}_boxplot.pdf")


#%%

for method in ["hilbert", "peaks"]:
    df = pd.read_csv(f"damping_{method}/damping_results.csv")
    
    dfm = df.merge(df_f)
    
    dfm["damping"] = -dfm["k"]/dfm["Freq"]
    
    fig, ax = plt.subplots(figsize=(15,6))
    
    sns.swarmplot(
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
    fig.savefig(f"bara/{method}_swarmplot.pdf")

#%%


directory = "damping_both"
os.makedirs(directory, exist_ok=True)
d = {}
methods = ["hilbert", "peaks"]
for method in methods:
    d[method] = pd.read_csv(f"damping_{method}/damping_results.csv")
    d[method] = d[method].rename(columns = {'k': 'k_'+method, 'q': 'q_'+method}) 
df = d['hilbert']
df = df.merge(d['peaks'])
df = df.merge(df_f)
for method in methods:
    df["damping_"+method] = - df["k_"+method] / df["Freq"]
    
trees = df["tree"].unique()    

for tree in trees:
    print(tree)
    dft = df[df["tree"]==tree]
    
    fig, ax = plt.subplots(1,2, sharey=True)
    
    sns.swarmplot(
        data=dft, 
        x="measurement", 
        y="damping_peaks", 
        hue="date",
        ax = ax[0], legend=False,
        )
    ax[0].grid()
    sns.swarmplot(
        data=dft, 
        x="measurement", 
        y="damping_hilbert", 
        hue="date",
        ax = ax[1], legend=False
        )
    ax[1].grid()
    plt.suptitle(f"Tree {tree}", )
    fig.savefig(f"{directory}/{tree}.pdf")

#%%

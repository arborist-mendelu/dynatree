#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:01:02 2023

@author: marik
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def get_data(probe):
    df = pd.read_csv("csv/solara_FFT.csv")
    if probe in ["Inclino(80)", "Inclino(81)"]:
    
        df_axes = {}
        for axis in ['X','Y']:
            df_axes[axis] = ( df[df["probe"]==f"{probe}{axis}"]
               .dropna(subset=['peaks', 'remark'], how='all')
               .drop(["remark","from","to"], axis=1)
               )
        df = pd.concat(df_axes.values())    
        df = df[["day","tree","measurement","peaks","probe"]]
    else:
    
        df = df[df["probe"]==probe]
        df = df[["day","tree","measurement","peaks","probe"]]
    
    
    df["peak"] = df["peaks"].str.split(" ", expand=True).iloc[:,1]
    df["leaves"] = "False"
    idx = df["day"].isin(["2021-06-29", "2022-08-16", "2023-07-17"])
    df.loc[idx,"leaves"] = "True"
    idx = df["day"].isin(["2024-04-10"])
    df.loc[idx,"leaves"] = "False & post reduction"
    df["freq"] = df[["peak"]].astype(float)
    df = df.dropna()
    return df

def plot_data(df,probe):
    
    fig, axs = plt.subplots(2,1,figsize=(14,10), sharex=True, sharey=True)
    ax = axs[0]
    # f_min = 0.1
    # delta_f_min = 0.05
    sns.swarmplot(
        data=df, 
        x="tree", 
        y="freq", 
        hue="day",
        ax = ax
        )
    [ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]
    ax.legend(ncol=3)
    ax.grid(alpha=0.4)
    ax.set(title=f"Základní frekvence {probe}")
    # ax.set(ylim=(0,14))
    
    ax = axs[1]
    sns.boxplot(
        data=df, 
        x="tree", 
        y="freq", 
        hue="leaves",
        ax = ax
        )
    [ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]
    ax.legend(loc=2, title="Leaves")
    ax.grid(alpha=0.4)
    
    axs[0].set(ylim=(0.1,0.5))
    axs[1].set(ylim=(0.1,0.5))
    return fig

fig = {}
for probe in ["Pt3", "Elasto(90)", "Inclino(81)", "Inclino(80)"]:
    df = get_data(probe)
    fig[probe] = plot_data(df, probe)
    
filename = "../outputs/fft_probes.pdf"
with PdfPages(filename) as pdf:
    for f in fig.values():
        pdf.savefig(f, bbox_inches='tight') 

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
    df = pd.read_csv("../outputs/FFT_csv_tukey.csv")
    
    days_with_leaves_true = ["2021-06-29", "2021-08-03", "2022-08-16", "2023-07-17", "2024-09-02"]
    days_after_first_reduction = ['2024-01-16', '2024-04-10', '2024-09-02']
    df = df[df["tree"] != "JD18"]

    # Set information about leaves.
    df.loc[:,"leaves"] = False
    idx = (df["day"].isin(days_with_leaves_true))
    df.loc[idx,"leaves"] = True
    
    df.loc[:,"reductionNo"] = 0
    idx = (df["day"].isin(days_after_first_reduction))
    df.loc[idx,"reductionNo"] = 1
    idx = (df["type"]=="afterro")
    df.loc[idx,"reductionNo"]=1
    idx = (df["type"]=="afterro2")
    df.loc[idx,"reductionNo"]=2
    df = df.dropna().sort_values(by="tree")
    
    df["state"] = df["leaves"].astype(str) + " & " + df["reductionNo"].astype(str)

    return df
   

def plot_data(df,probe):
    
    fig, axs = plt.subplots(2,1,figsize=(14,10), sharex=True, sharey=True)
    ax = axs[0]
    # f_min = 0.1
    # delta_f_min = 0.05
    sns.swarmplot(
        data=df, 
        x="tree", 
        y="peak", 
        hue="state",
        ax = ax, 
        s=4
        )
    [ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]
    ax.legend(title="Leaves")
    ax.grid(alpha=0.4)
    ax.set(title=f"Základní frekvence {probe}")
    # ax.set(ylim=(0,14))
    
    ax = axs[1]
    sns.boxplot(
        data=df, 
        x="tree", 
        y="peak", 
        hue="state",
        ax = ax
        )
    [ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]
    ax.legend(loc=2, title="Leaves & No of reductions")
    ax.grid(alpha=0.4)
    
    # axs[0].set(ylim=(0.1,0.5))
    # axs[1].set(ylim=(0.1,0.5))
    return fig

fig = {}
for probe in ["Elasto(90)","blueMaj","yellowMaj"]:
    df = get_data(probe)
    fig[probe] = plot_data(df, probe)
    
filename = "../outputs/fft_boxplots_for_probes_tukey.pdf"
with PdfPages(filename) as pdf:
    for f in fig.values():
        pdf.savefig(f, bbox_inches='tight') 

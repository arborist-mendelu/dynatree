#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:01:02 2023

@author: DYNATREE project, ERC CZ no. LL1909 "Tree Dynamics: Understanding of Mechanical Response to Loading"
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import config

# df_long['diff'] = df_long['ori']-df_long['peak']
#%%

def get_data():
    # fix by manual peaks
    df_fix = pd.read_csv(config.file["FFT_manual_peaks"])
    df_fix["type"] = df_fix["measurement_type"]
    df_fix['peak'] = df_fix['peaks'].str.strip()  # Odstranění vedoucí mezery
    df_fix['peak'] = df_fix['peak'].str.split().str[0]  # Rozdělení a ponechání první části
    df_fix['peak'] = df_fix['peak'].astype(float)  # Převedení na float
    df_fix = df_fix.drop(['peaks'], axis=1)
    
    df_long = pd.read_csv(config.file["outputs/FFT_csv_tukey"])
    df_fix = df_fix.set_index(['type', 'day', 'tree', 'measurement', 'probe'])
    df_long = df_long.set_index(['type', 'day', 'tree', 'measurement', 'probe'])
    # df_long['ori'] = df_long['peak']
    df_long.update(df_fix['peak'])
    df_long = df_long.reset_index()


    df = df_long.pivot(index=["type","day","tree","measurement"], columns="probe", values="peak")
    df = df.reset_index()
    # df ["diff1"] = df["Elasto(90)"] - df["blueMaj"] 
    # df ["diff2"] = df["Elasto(90)"] - df["yellowMaj"] 
    
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
    df = df.sort_values(by="tree")
   
    
   
    df["state"] = df["leaves"].astype(str) + " & " + df["reductionNo"].astype(str)
    df = df.sort_values(by=['tree','reductionNo', 'leaves'], ascending=[True, False, True]).reset_index(drop=True)

    return df
   

def plot_data(df,probe):
    
    fig, axs = plt.subplots(2,1,figsize=(14,10), sharex=True, sharey=True)
    ax = axs[0]
    # f_min = 0.1
    # delta_f_min = 0.05
    sns.swarmplot(
        data=df, 
        x="tree", 
        y=probe, 
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
        y=probe, 
        hue="state",
        ax = ax
        )
    [ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]
    ax.legend(loc=2, title="Leaves & No of reductions")
    ax.grid(alpha=0.4)
    
    axs[0].set(ylim=(0.1,0.6))
    axs[1].set(ylim=(0.1,0.6))
    return fig

df = get_data()

fig = {}
for probe in ["Elasto(90)","blueMaj","yellowMaj","Pt3","Pt4","a01_z","a01_z","a02_z","a03_z","a04_z"]:
    fig[probe] = plot_data(df, probe)
    
filename = "../outputs/fft_boxplots_for_probes_tukey.pdf"
with PdfPages(filename) as pdf:
    for f in fig.values():
        pdf.savefig(f, bbox_inches='tight') 

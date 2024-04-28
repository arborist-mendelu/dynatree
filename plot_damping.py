#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:36:33 2024

@author: marik
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("damping_output/damping_results.csv")

df = df.dropna()
df["utlum"] = df[["hilbert","peaks","wavelet"]].mean(axis=1)
df['A'] = np.abs(df["hilbert"]/df["utlum"] - 1)
df['B'] = np.abs(df["peaks"]/df["utlum"] - 1)
df['C'] = np.abs(df["wavelet"]/df["utlum"] - 1)

def listy(datum):
     if datum in ["2021-06-29", "2022-08-16"]:
         return True
     return False
 
df['listy'] = df['date'].map(listy)
df_all = df
df_bad = df[(df[['A', 'B', 'C']].max(axis=1) > 0.2)]
df_fine = df[(df[['A', 'B', 'C']].max(axis=1) < 0.2)]


for d,t,name in zip([df, df_fine], 
               ["Všechna měření kromě zcela zkažených", 
                "Měření, kde všechny tři metody dávají podobné výsledky"
                ],
               ["all", "great"]
               ):
    fig, axs = plt.subplots(2,1,figsize=(10,6), sharex=True)
    for funkce,ax in zip([sns.swarmplot, sns.boxplot],axs) :
        funkce(
            data=d, 
            x="tree", 
            y="utlum", 
            hue="listy",
            ax = ax
            )
        [ax.axvline(x+.5,color='gray', lw=1.5) for x in ax.get_xticks()]
        ax.legend(loc=2)
        ax.grid(alpha=0.4)
        ax.set(ylim=(0.1,0.6))
        ax.legend(title="Olistění")
    plt.suptitle(t)
    plt.tight_layout()
    plt.savefig(f"damping_output/00_ouput_{name}.pdf")

df_bad.drop(columns=['A', 'B', 'C']).to_csv("damping_output/00_bad.csv", index=False)


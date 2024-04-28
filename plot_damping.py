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

    
fig, ax = plt.subplots(figsize=(10,6))

sns.swarmplot(
    data=df, 
    x="tree", 
    y="utlum", 
    hue="listy",
    ax = ax
    )
[ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]
ax.legend(loc=2)
ax.grid(alpha=0.4)
ax.set(title="Tlumení, všechna měření kromě zcela zkažených", ylim=(0.1,0.6))



df_bad = df[(df[['A', 'B', 'C']].max(axis=1) > 0.2)]

fig, ax = plt.subplots(figsize=(10,6))

df = df[(df[['A', 'B', 'C']].max(axis=1) < 0.1)]

sns.boxplot(
    data=df, 
    x="tree", 
    y="utlum", 
    hue="listy",
    ax = ax
    )
[ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]
ax.legend(loc=2)
ax.grid(alpha=0.4)
ax.set(title="Tlumení, měření, kde všechny metody dávají podobné výsledky", ylim=(0.1,0.6))

# plt.savefig("outputs/swarmplot.pdf")


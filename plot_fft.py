#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 19:01:02 2023

@author: marik
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

probe = "Pt4"
df = pd.read_csv("results/fft.csv", dtype = {'tree': str, 'date': str})
df = df[df["probe"].str.contains(probe)]
df["leaves"] = False
idx = (df["date"] == "2021-06-29") | (df["date"] == "2022-08-16")
df.loc[idx,"leaves"] = True
#%%

df.reset_index(inplace=True)

# %% boxplot

# fig, ax = plt.subplots(figsize=(15,10))
# sns.boxplot(
#     data=df[(df["Freq"]>f_min) & (df["Delta freq"]<delta_f_min)], 
#     x="tree", 
#     y="Freq", 
#     #style="tree", 
#     hue="date", 
#     palette=sns.color_palette("tab10")[:3],
#     #s = 100,
#     ax=ax
#     )
# # ax.set(ylim=(.16,.38))
# [ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]
# ax.legend(loc=2)
# ax.grid(alpha=0.4)
# ax.set(title="Základní frekvence")

# %% swarmplot

fig, axs = plt.subplots(2,1,figsize=(14,10), sharex=True)

ax = axs[0]

f_min = 0.1
delta_f_min = 0.05
sns.swarmplot(
    data=df[(df["freq"]>f_min) & (df["err"]<delta_f_min)], 
    x="tree", 
    y="freq", 
    hue="date",
    ax = ax
    )
[ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]
ax.legend(loc=2)
ax.grid(alpha=0.4)
ax.set(title=f"Základní frekvence {probe}")
# ax.set(ylim=(0,14))

ax = axs[1]

f_min = 0.15
delta_f_min = 0.05

df2 = df.copy()[(df["freq"]>f_min) & (df["err"]<delta_f_min)]

# fig, ax = plt.subplots(figsize=(10,6))

sns.boxplot(
    data=df2, 
    x="tree", 
    y="freq", 
    hue="leaves",
    ax = ax
    )
[ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]
ax.legend(loc=2, title="Leaves")
ax.grid(alpha=0.4)
# ax.set(title=f"Základní frekvence (dvě měření s listy a dvě bez listů) {probe}")
# ax.set(ylim=(0,14))
plt.savefig("outputs/boxplot.pdf")

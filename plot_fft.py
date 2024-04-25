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

def uprav_df(df):
    df["tree"] = [i.split("_")[0] for i in df.index]
    return df
colors = {'1': 'blue', '2': 'red'}
dfs = [
     pd.read_excel(f"fft_data_01_Mereni_Babice_{i}_optika_zpracovani.xlsx", index_col=0)
     for i in ["22032021","29062021","05042022","16082022"]
     ]

dfs[0]['date'] = "2021-03-22"
dfs[1]['date'] = "2021-06-29"
dfs[2]['date'] = "2022-04-05"
dfs[3]['date'] = "2022-08-16"

dfs[0]['listy'] = False
dfs[1]['listy'] = True
dfs[2]['listy'] = False
dfs[3]['listy'] = True


df = pd.concat(dfs)

df = df.copy().pipe(uprav_df)

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

fig, ax = plt.subplots(figsize=(10,6))

f_min = 0.1
delta_f_min = 0.05
sns.swarmplot(
    data=df[(df["Freq"]>f_min) & (df["Delta freq"]<delta_f_min)], 
    x="tree", 
    y="Freq", 
    hue="date",
    ax = ax
    )
[ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]
ax.legend(loc=2)
ax.grid(alpha=0.4)
ax.set(title="Základní frekvence")
plt.savefig("outputs/swarmplot.pdf")

df[["tree","date","index","Freq"]].sort_values(by=["tree","date","Freq"]).to_csv("outputs/FFT_freq.csv", index=None, header=False)

# %% s listim/bez listi

f_min = 0.15
delta_f_min = 0.05

df2 = df.copy()[(df["Freq"]>f_min) & (df["Delta freq"]<delta_f_min)]

fig, ax = plt.subplots(figsize=(10,6))

sns.boxplot(
    data=df2, 
    x="tree", 
    y="Freq", 
    hue="listy",
    ax = ax
    )
[ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]
ax.legend(loc=2, title="Olistění")
ax.grid(alpha=0.4)
ax.set(title="Základní frekvence (dvě měření s listy a dvě bez listů)")
plt.savefig("outputs/boxplot.pdf")

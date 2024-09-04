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
import statsmodels.api as sm
import scipy


df = pd.read_csv("csv/solara_FFT.csv")
df = df[df["probe"]=='Elasto']
df["peak"] = df["peaks"].str.split(" ", expand=True).iloc[:,1]
df["date"] = df["day"]
df = df[["date","tree","measurement","peak"]]
#%%
df["leaves"] = False
idx = (df["date"] == "2021-06-29") | (df["date"] == "2022-08-16") | (df["date"] == "2023-07-17")
df.loc[idx,"leaves"] = True
df["freq"] = df[["peak"]].astype(float)
#%%
df = df.dropna()
#%%

probe = "Elasto"
fig, axs = plt.subplots(2,1,figsize=(14,10), sharex=True, sharey=True)

ax = axs[0]

f_min = 0.1
delta_f_min = 0.05
sns.swarmplot(
    data=df, #[(df["freq"]>f_min) & (df["err"]<delta_f_min)], 
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

# f_min = 0.15
# delta_f_min = 0.05

# df2 = df.copy()[(df["freq"]>f_min) & (df["err"]<delta_f_min)]

# fig, ax = plt.subplots(figsize=(10,6))

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

# ax.set(title=f"Základní frekvence (dvě měření s listy a dvě bez listů) {probe}")
# ax.set(ylim=(0,14))
plt.savefig("outputs/boxplot.pdf")


#%%

df = df.reset_index(drop=True)

#%%

# ANOVA, t-test
#
# We start with transformation, since each tree has its own frequency.

trees = df['tree'].pipe(np.unique)

for tree in trees:
    idx = df["tree"]==tree
    mean = df.loc[idx,["freq"]].mean().values
    std = df.loc[idx,["freq"]].std().values
    df.loc[idx,["freq_diff_from_mean_rescaled"]] = (df.loc[idx,"freq"] - mean) / std

#%%

sm.qqplot(df[''], line='45', fit = True) 
plt.show()

#%%

# https://gist.github.com/robert-marik/635affe37158d3fae1ef4f5bf3798dd8
skupiny = [df.loc[i,'freq_diff_from_mean_rescaled'] for i in df.groupby(by='leaves').groups.values()]
#print(skupiny) # kontrola prvku ve skupine
#print(len(skupiny)) # kontrola poctu skupin
for i in skupiny:
    print(np.mean(i), np.std(i))

print("ANOVA: ", scipy.stats.f_oneway(*skupiny))
print("t-test: ", scipy.stats.ttest_ind(*skupiny, equal_var=True))

#%%


for leaves in df['leaves'].unique():
  sm.qqplot(df[df['leaves']==leaves].loc[:,'freq_diff_from_mean_rescaled'], line='45', fit = True)
  ax = plt.gca()
  ax.set_title(f"Leaves {leaves}")

#%%
# for tree in df['tree'].unique():
#   sm.qqplot(df[df['tree']==tree].loc[:,'freq_diff_from_mean_rescaled'], line='45', fit = True)
#   ax = plt.gca()
#   ax.set_title(f"Tree {tree}")

#%%

skupiny = [df.loc[i,'freq_diff_from_mean_rescaled'] for i in df.groupby(by='tree').groups.values()]
#print(skupiny) # kontrola prvku ve skupine
#print(len(skupiny)) # kontrola poctu skupin
print("ANOVA: ", scipy.stats.f_oneway(*skupiny))

#%%
sns.boxplot(df, x="leaves", y="freq_diff_from_mean_rescaled")

#%%
sns.boxplot(df, x="tree", y="freq_diff_from_mean_rescaled")

#%%
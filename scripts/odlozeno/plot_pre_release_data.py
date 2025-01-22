#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

NEAKTUALNI


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
df1 = pd.read_excel("outputs/pre_release_data_01_Mereni_Babice_22032021_optika_zpracovani.xlsx", index_col=0)
df2 = pd.read_excel("outputs/pre_release_data_01_Mereni_Babice_29062021_optika_zpracovani.xlsx", index_col=0)
df3 = pd.read_excel("outputs/pre_release_data_01_Mereni_Babice_05042022_optika_zpracovani.xlsx", index_col=0)
df1['date'] = "22-03-2021"
df2['date'] = "29-06-2021"
df3['date'] = "05-04-2022"
df = pd.concat([df1,df2,df3])

df = df.copy().pipe(uprav_df)

#df1.plot(x="Elasto(90)", y="Force(100)", by="tree")
fig, ax = plt.subplots(figsize=(15,10))
sns.scatterplot(
    data = df, 
    x="Elasto(90)", 
    y="Force(100)", 
    style="tree", 
    hue="date", 
    palette=sns.color_palette("tab10")[:3],
    s = 100,
    ax=ax
    )
# %%

fig, ax = plt.subplots(figsize=(15,10))
df['test'] = np.sqrt(df["PT3_UX0"]**2 + df["PT3_UY0"]**2)/np.sqrt(df["Inclino(80)X"]**2+df["Inclino(80)Y"]**2)
sns.violinplot(
    data = df, 
    x="tree", 
    y="test", 
    #style="tree", 
    hue="date", 
    palette=sns.color_palette("tab10")[:3],
    #s = 100,
    ax=ax
    )
ax.set(ylim=(-500,1000))
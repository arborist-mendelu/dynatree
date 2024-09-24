#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 20:47:57 2024

@author: marik
"""

import pandas as pd

def add_leaves_info(df_):
    df = df_.copy()
    days_with_leaves_true = ["2021-06-29", "2021-08-03", "2022-08-16", "2023-07-17", "2024-09-02"]
    days_after_first_reduction = ['2024-01-16', '2024-04-10', '2024-09-02']
    
    # Set information about leaves.
    df.loc[:,"leaves"] = False
    idx = (df["day"].isin(days_with_leaves_true))
    df.loc[idx,"leaves"] = True
    # no reduction is default
    df.loc[:,"reductionNo"] = 0
    # one reduction: afterro or any day later
    idx = (df["day"].isin(days_after_first_reduction))
    df.loc[idx,"reductionNo"] = 1
    idx = (df["type"]=="afterro")
    df.loc[idx,"reductionNo"]=1
    # two reductions
    idx = (df["type"]=="afterro2")
    df.loc[idx,"reductionNo"]=2
    return df

df = pd.read_csv("../outputs/anotated_regressions_static.csv", index_col=0)
df = df.dropna(subset=["Independent","Dependent"],how='all')
df = df[df["lower_cut"]==0.3]
df = df.dropna(how='all', axis=0)
df = df[~df['Dependent'].str.contains('Min')]
df = df[~df['tree'].str.contains('JD')]
df = df[~ df["failed"]]
df = df.drop(["Intercept","p-value","stderr","intercept_stderr","lower_cut", "upper_cut"], axis=1)

a = df[(df["optics"]) & (df['Dependent'].str.contains('Pt'))]
b = df[~df["optics"]]
oridf = pd.concat([a,b]).reset_index(drop=True)

#%%

df = oridf.copy()
df = df[
    (df["measurement"]=="M01") & 
     ((df["Independent"]=="M") |
      (df["Independent"]=="M_Elasto"))
    ].drop(["optics"], axis=1)


# Nejprve vytáhneme hodnoty Slope pro pull=0
df_zero_pull = df[df['pullNo'] == 0].copy()

# Přejmenujeme sloupec Slope, aby bylo jasné, že jde o referenční hodnoty
df_zero_pull = df_zero_pull.rename(columns={'Slope': 'Slope_zero_pull'})


df_nonzero_pull = df[df['pullNo'] != 0].copy()
# df_nonzero_pull


# Merge původního DataFrame s tím, kde je pull=0, na základě společných sloupců
df_merged = pd.merge(df_nonzero_pull, df_zero_pull[['day', 'tree', 'Independent', 'Dependent', 'type','Slope_zero_pull']],
                     on=['day', 'tree', 'Independent', 'Dependent',"type"], how='left')

# Vydělení hodnoty Slope referenční hodnotou Slope_zero_pull
for i in ['Slope']:
    df_merged[f'{i}_normalized'] = df_merged[i] / df_merged[f'{i}_zero_pull']
df_merged = df_merged.drop(columns=["reason"])
df_merged = df_merged.dropna()


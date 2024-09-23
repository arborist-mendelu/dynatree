#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 10:20:56 2024

@author: marik
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lib_dynatree
from scipy.stats import ttest_1samp
from scipy import stats


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
oridf

# # Je u prvního měření větší tuhost?

df = oridf.copy()
df = df[
    (df["measurement"]=="M01") & 
     (df["Independent"]=="M")
    ].drop(["measurement","optics"], axis=1)
df

# +
# Nejprve vytáhneme hodnoty Slope pro pull=0
df_zero_pull = df[df['pullNo'] == 0].copy()

# Přejmenujeme sloupec Slope, aby bylo jasné, že jde o referenční hodnoty
df_zero_pull = df_zero_pull.rename(columns={'Slope': 'Slope_zero_pull'})

# Merge původního DataFrame s tím, kde je pull=0, na základě společných sloupců
df_merged = pd.merge(df, df_zero_pull[['day', 'tree', 'Independent', 'Dependent', 'Slope_zero_pull']],
                     on=['day', 'tree', 'Independent', 'Dependent'], how='left')

# Vydělení hodnoty Slope referenční hodnotou Slope_zero_pull
for i in ['Slope']:
    df_merged[f'{i}_normalized'] = df_merged[i] / df_merged[f'{i}_zero_pull']
# -

df_merged

target = "Slope"
subdf = df_merged[df_merged["pullNo"]!=0].loc[:,["pullNo","Slope_normalized","tree"]]
fig, ax = plt.subplots(1,1,figsize=(20,5))
subdf['pullNo'] = subdf['pullNo'].astype('category')
sns.stripplot(data=subdf, x='tree', y=f'{target}_normalized', hue='pullNo', ax=ax, jitter=0.4)
ax.grid()
ax.set(ylim=(None,1.5))

# +
result1 = subdf.groupby('tree')['Slope_normalized'].agg(['mean', 'std']).reset_index()

# Funkce pro testování hypotézy
def test_hypothesis(group):
    t_stat, p_value = ttest_1samp(group['Slope_normalized'], 1)
    return pd.Series({'t_stat': t_stat, 'p_value': p_value})

# Aplikace funkce na skupiny dle kategorie 'tree'
result2 = subdf.groupby('tree').apply(test_hypothesis, include_groups=False).reset_index()

result = pd.merge(result1,result2, on=["tree"])
result["H0_rejected"] = result["p_value"]<0.05
result.sort_values(by="p_value").reset_index(drop=True)
# -

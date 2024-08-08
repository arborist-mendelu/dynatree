#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:46:13 2024

@author: marik
"""

import pickle
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

files = glob.glob("temp_figs/*pkl")

keys = []
for file in files:
    with open(file,'rb') as f:
        data = pickle.load(f)
    keys = keys + [i for i in data.keys() if not isinstance(i,tuple)] + ["_".join(i) for i in data.keys() if isinstance(i, tuple)]
keys = np.unique(keys)


#%%

df_dict = {}
i = 0
for file in files:
    print (file)
    with open(file,'rb') as f:
        data = pickle.load(f)
    for k in data.keys():
        if isinstance(k, tuple):
            name = "_".join(k)
        else: 
            name = k
        if k == 'data':
            continue
        if data[k] is not None:
            i = i+1
            d = data['data']
            df_dict[i] = [d['date'], d['tree'], d['measurement'], name, data[k]['peak_position']]

#%%

df = pd.DataFrame(df_dict).T
df.columns = ["date","tree", "measurement","device","freq"]
df.to_csv("temp_figs/peaks.csv", header=False, index=False)

#%%

import seaborn as sns

devices =df['device'].unique()
devices

fix, ax = plt.subplots(3,2,sharex=True, sharey=True, figsize=(15,10))
ax = ax.reshape(-1)

for i,d in enumerate(devices):
    ax[i].set(title=d)
    sns.scatterplot(df[df['device']==d], ax = ax[i])
    # ax[i].set(ylim=(0,1))
plt.tight_layout()

#%%
df_wide = df.pivot(index=df.columns[:3], columns='device', values='freq').reset_index().sort_values(by=["tree","date","measurement"]).reset_index(drop=True)
df_wide
df_wide.to_excel('freq_acc.xlsx')

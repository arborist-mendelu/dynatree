#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:24:56 2024

@author: marik
"""

import lib_dynatree 
import lib_dynasignal
import lib_find_measurements
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib
import lib_FFT


df = lib_find_measurements. get_all_measurements(method='all', type='all')
df = df[df['measurement'] != 'M01']

#%%

# day, tree, measurement, measurement_type = "2024-09-02", "BK01", "M05", "normal"

# m = lib_dynatree.DynatreeMeasurement(
#     day=day, tree=tree, measurement=measurement, measurement_type=measurement_type
#     )


end=100
end=10000

def do_welch_spectra(m):
    fig, axs = plt.subplots(2,1, figsize=(12,6))
    dt = 0.0002
    probe = "a03_z"

    sig = lib_FFT.DynatreeSignal(m, probe)

    ax = axs[0]    
    sig.signal_full.plot(ax=ax)
    sig.signal.plot(ax=ax)
    ax.set(ylim=(sig.signal.min(), sig.signal.max()))
    ax.set(title = f"{m.day} {m.tree} {m.measurement} {probe}")

    ax = axs[1]
    ans = lib_dynasignal.do_welch(pd.DataFrame(sig.signal),  nperseg=2**8)
    ans.plot(ax=ax)
    ax.set(yscale='log')    
    ax.grid()
    # data = m.data_acc5000
    # cols = [i for i in data.columns if "z" in i]
    # data = data.loc[:,cols]
    
    # a = data.loc[:,'a03_z'].copy()
    # a.loc[:20] = 0
    # release = a.idxmax()
    # data = data.loc[release:,:]
    # data = data - data.mean()
    
    # length = data.index[-1]-data.index[0]
    # if length < 60:
    #     df_ = pd.DataFrame(0, columns=data.columns,
    #                       index = np.arange(0,60-length,dt)+data.index[-1]+dt    
    #                       )
    
    #     data = pd.concat([data,df_])

    # tukey_window = signal.windows.tukey(len(data), alpha=tukey_alpha, sym=False)
    # data = data.mul(tukey_window, axis=0)
    # ax = axs[0]
    # data.plot(ax=ax)
    # ax.set(title = str(m).replace("Dynatree measurement",""), xlabel="Time / s", 
    #        ylabel = "Acceleration / (m*s^-2)")
    # ax.grid()

    # ax = axs[1]
    # welch = lib_dynasignal.do_welch(data, nperseg=2**15)
    # welch.loc[1:end,:].plot(ax=ax)
    # ax.set(yscale='log')
    # ax.set(xlabel="Freq", ylabel="Power spectral density")
    # ax.grid()
    return fig,axs

#%%

matplotlib.use('Agg')
pbar = tqdm(total=len(df))
for i,row in df.iterrows():
    m = lib_dynatree.DynatreeMeasurement(day=row['day'], tree=row['tree'], measurement=row['measurement'], measurement_type=row['type'])
    pbar.set_description(f"{m}")
    fig, ax = do_welch_spectra(m)
    filename = f"../temp/welch/{m.tree}_{m.measurement_type}_{m.day}_{m.measurement}.png"
    fig.savefig(filename)
    pbar.update(1)
    plt.close('all')
    break
pbar.close()
    
    
#%%

# do_welch_spectra(m)    
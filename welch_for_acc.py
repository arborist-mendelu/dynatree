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

pbar.close()
    
    
#%%

# do_welch_spectra(m)    
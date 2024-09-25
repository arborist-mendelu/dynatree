#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:24:56 2024

@author: marik
"""

import lib_dynatree 
import lib_dynasignal
import lib_find_measurements
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib
import lib_FFT

df = lib_find_measurements. get_all_measurements(method='all', type='all')
df = df[df['measurement'] != 'M01']

def do_welch_spectra(row):
    
    measurement_type, day, tree = row
    measurements = df[
        (df["type"]==row['type']) & 
        (df["day"]==row['day']) &  
        (df["tree"]==row['tree']) 
        ].loc[:,"measurement"].drop_duplicates().tolist()

    fig, axs = plt.subplots(2,1, figsize=(12,6))
    dt = 0.0002
    probe = "a03_z"
    lb = 0
    ub = 0

    for measurement in measurements:
        m = lib_dynatree.DynatreeMeasurement(
            day=day, 
            tree=tree, measurement=measurement, 
            measurement_type=measurement_type
        )
        sig = lib_FFT.DynatreeSignal(m, probe)
        lb = min(lb, sig.signal.min())
        ub = max(ub, sig.signal.max())

        ax = axs[0]    
        # sig.signal_full.plot(ax=ax)
        sig.signal.plot(ax=ax, alpha=0.5)
    
        ax = axs[1]
        ans = lib_dynasignal.do_welch(pd.DataFrame(sig.signal),  nperseg=2**8)
        ans.plot(ax=ax)
        
    axs[0].set(ylim=(lb,ub))
    axs[0].set(title = f"{m.day} {m.tree} {m.measurement} {probe}")
    axs[1].set(yscale='log')    
    axs[1].grid()
    axs[1].legend(measurements)
    
    return fig,axs

#%%



matplotlib.use('Agg')
tdf = df[["type","day","tree"]].copy().drop_duplicates()
pbar = tqdm(total=len(tdf))

for i,row in tdf.iterrows():
    fig, ax = do_welch_spectra(row)
    filename = f"../temp/welch/{row['tree']}_{row['type']}_{row['day']}.png"
    fig.savefig(filename)
    pbar.update(1)
    plt.close('all')

pbar.close()
    
    
#%%

# do_welch_spectra(m)    
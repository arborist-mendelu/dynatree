#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:24:56 2024

@author: marik
"""

from dynatree import dynasignal, dynatree, FFT, find_measurements
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib
import config

df = find_measurements.get_all_measurements(method='all', type='all')
df = df[df['measurement'] != 'M01']

probe = "a03_z"

failed_df = pd.read_csv(config.file["FFT_failed"])
failed = failed_df[failed_df["probe"]==probe].drop(columns=["probe"]).values.tolist()

def do_welch_spectra(row):
    
    measurement_type, day, tree = row
    measurements = df[
        (df["type"]==row['type']) & 
        (df["day"]==row['day']) &  
        (df["tree"]==row['tree']) 
        ].loc[:,"measurement"].drop_duplicates().tolist()

    fig, axs = plt.subplots(2,1, figsize=(12,6))
    dt = 0.0002
    lb = 0
    ub = 0

    for measurement in measurements:
        m = dynatree.DynatreeMeasurement(
            day=day, 
            tree=tree, measurement=measurement, 
            measurement_type=measurement_type
        )
        if [measurement_type, day, tree, measurement] in failed:
            #print(f"Skipping {measurement_type} {day} {tree} {measurement}")
            continue
        sig = FFT.DynatreeSignal(m, probe)
        lb = min(lb, sig.signal.min())
        ub = max(ub, sig.signal.max())

        ax = axs[0]    
        # sig.signal_full.plot(ax=ax)
        sig.signal.plot(ax=ax, alpha=0.5)
    
        ax = axs[1]
        ans = dynasignal.do_welch(pd.DataFrame(sig.signal), nperseg=2 ** 8)
        ans.plot(ax=ax)
        
    axs[0].set(ylim=(lb,ub), ylabel="Acceleration", xlabel="Time / s")
    axs[0].set(title = f"{m.day} {m.tree} {m.measurement_type} {probe}")
    axs[1].set(yscale='log', ylabel="Power spectral density", xlabel="Freq / Hz")    
    axs[1].grid()
    axs[1].legend(measurements)
    plt.tight_layout()
    
    return fig,axs

#%%



matplotlib.use('Agg')
tdf = df[["type","day","tree"]].copy().drop_duplicates()
pbar = tqdm(total=len(tdf))

for i,row in tdf.iterrows():
    fig, ax = do_welch_spectra(row)
    filename = f"../temp/welch/{row['tree']}_{row['day']}_{row['type']}.png"
    fig.savefig(filename)
    pbar.update(1)
    plt.close('all')

pbar.close()
    
    
#%%

# do_welch_spectra(m)    
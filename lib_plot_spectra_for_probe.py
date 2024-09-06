#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:22:45 2024

Plot spectra from one probe. Data from csv/solara_FFT.csv

@author: marik
"""

import lib_dynatree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

def plot_spectra_for_all_probes(
    measurement_type = "normal",
    day = "2021-03-22",
    tree = "BK11",
    measurement = "M02",
    fft_results = None,
    DT = 0.01):

    data = lib_dynatree.DynatreeMeasurement(day, tree, measurement, measurement_type=measurement_type)
    
    if fft_results is None:
        fft_results = pd.read_csv("csv/solara_FFT.csv", index_col=[0,1,2,3,4])
    
    try:
        subset_fft_results = fft_results.loc[(measurement_type, day, tree, measurement, slice(None)),:]
    except:
        return None
    
    subset_fft_results = subset_fft_results.dropna(subset=['peaks', 'remark'], how='all')
    idx = subset_fft_results.index
    
    figs = {}
    remarks = {}
    for i,row in subset_fft_results.iterrows():
        probe = i[-1]
        if probe[:2] == "a0":
            dataset = data.data_acc
        elif probe[:6] in  ["Elasto", "Inclin"]:
            dataset = data.data_pulling
        else:
            dataset = data.data_optics_pt34
        
        dataset = dataset.loc[:,[probe]].interpolate()
        
        fig, ax = plt.subplots(2,1)
        try:
            if row['to'] < 0.001:
                row['to'] = dataset.index[-1]
            dataset.loc[row['from']:row['to'],:].plot(ax=ax[0], legend=None)
            time_fft =  np.arange(row['from'],row['to'],DT)
            N = time_fft.shape[0]  # get the number of points
            xf_r = fftfreq(N, DT)[:N//2]
            signal_fft = np.interp(time_fft, dataset.index, dataset[probe].values.reshape(-1))
        
            time_fft = time_fft - time_fft[0]
            signal_fft = signal_fft - np.nanmean(signal_fft) # mean value to zero
            yf = fft(signal_fft)  # preform FFT analysis
            yf_r = 2.0/N * np.abs(yf[0:N//2])
            upper_bound = np.max(yf_r)
            ax[1].plot(xf_r,yf_r,".",color="C1")
            
            if not pd.isna(row['peaks']):
                for freq in row['peaks'].split(" "):
                    if freq == "":
                        continue
                    ax[1].axvline(x=float(freq), color='r', linestyle='--')
        
            ax[1].set(xlim=(0,3), yscale="log", ylim=(upper_bound/10**4, upper_bound*6))
            ax[1].grid()
            
            ax[0].set(title=f"{measurement_type} {day} {tree} {measurement} {probe}")
        except:
            pass
        
        figs[probe] = {'fig':fig, 'remark':"" if pd.isna(row['remark']) else row['remark'], 'peaks': row['peaks']}
        # plt.close(fig)
            
        
    return figs

if __name__ == "__main__":
    figs = plot_spectra_for_all_probes()
    for i in figs.values():
        plt.show(i['fig'])
        print(i['remark'])
    



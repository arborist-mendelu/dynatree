#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 14:22:45 2024

Plot spectra from one probe. Data from csv/solara_FFT.csv

@author: marik
"""

import dynatree.dynatree as dynatree
import dynatree.find_measurements as find_measurements
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib
from tqdm import tqdm

@dynatree.timeit
def plot_spectra_for_all_probes(
    measurement_type = "normal",
    day = "2021-03-22",
    tree = "BK11",
    measurement = "M02",
    fft_results = None,
    DT = 0.01,
    log_x = False,
    xmax = 3
    ):

    if measurement == "M01":
        print("M01 is not considered")
        return None
    data = dynatree.DynatreeMeasurement(
        day=day, 
        tree=tree, 
        measurement=measurement,
        measurement_type=measurement_type)
    
    if fft_results is None:
        fft_results = pd.read_csv("../csv/solara_FFT.csv", index_col=[0, 1, 2, 3, 4])
    
    try:
        subset_fft_results = fft_results.loc[(measurement_type, day, tree, measurement, slice(None)),:]
    except Exception as e:
        print(f"Exception {e}")
        return None
    
    subset_fft_results = subset_fft_results.dropna(subset=['peaks', 'remark'], how='all')
    idx = subset_fft_results.index
    
    figs = {}
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
        
            if log_x:
                xmin = time_fft[1]
            else:
                xmin = 0
            ax[1].set(xlim=(xmin,xmax), yscale="log", ylim=(upper_bound/10**4, upper_bound*6))
            if log_x:
                ax[1].set(xscale="log")
            ax[1].grid()
            
            ax[0].set(title=f"{measurement_type} {day} {tree} {measurement} {probe}")
        except:
            pass
        
        figs[probe] = {
            'fig':fig, 
            'remark':"" if pd.isna(row['remark']) else row['remark'], 
            'peaks': row['peaks'],
            'probe': probe
            }
        # plt.close(fig)
            
        
    return figs

def main():
    # measurement_type = "normal"
    # day = "2021-03-22"
    # tree = "BK11"
    # measurement = "M03"

    # figs = plot_spectra_for_all_probes(
    #     measurement_type=measurement_type,
    #     day=day,
    #     tree=tree,
    #     measurement=measurement)
    
    # return
    try:
        matplotlib.use('TkAgg') # https://stackoverflow.com/questions/39270988/ice-default-io-error-handler-doing-an-exit-pid-errno-32-when-running
    except:
        matplotlib.use('Agg')
    all = lib_find_measurements.get_all_measurements(method='all', type='normal',)
    pbar = tqdm(total=len(all))
    for i,row in all.iterrows():
        if row['measurement'] == "M01":
            continue
        # print (f"{row['date']} {row['tree']} {row['measurement']}")
        pbar.set_description(f"{row['type']} {row['date']} {row['tree']} {row['measurement']}")
        figs = plot_spectra_for_all_probes(
            measurement_type=row['type'],
            day=row['day'],
            tree=row['tree'],
            measurement=row['measurement'])
        # for i in figs.values():
            # plt.show(i['fig'])
            # print(i['remark'])
        if figs is None:
            continue
        for f in figs.values():
            f['fig'].text(0.01,0.01,f['remark'][:100])
            f['fig'].text(0.4,0.4,f"peaks {f['peaks']}")
            f['fig'].savefig(f"../temp_spectra/{row['type']}_{row['day']}_{row['tree']}_{row['measurement']}_{f['probe'].replace('(','').replace(')','')}.pdf")
        plt.close('all')
        pbar.update(1)
    pbar.close()    
    
if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:24:29 2024

@author: marik
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import pywt

from lib_dynatree import get_data, get_all_measurements

date2color = {"2021-03-22": "C0", "2021-06-29": "C1", "2022-04-05": "C2", 
             "2022-08-16": "C3"}

def get_limits(date, tree, measurement, csvdir="csv"):
    """
    Get limits for damping determinantion. The limits are taken from file
    csv/oscilation_times_remarks.csv from the coresponding columns. 
    If the limits for damping are not given, limits for fft are used.
    """
    df_remarks = pd.read_csv(f"{csvdir}/oscillation_times_remarks.csv")
    bounds_for_oscillations = df_remarks[
        (df_remarks["tree"]==f"BK{tree}") &
        (df_remarks["measurement"]==f"M0{measurement}") &
        (df_remarks["date"]==date)
        ]    
    end = bounds_for_oscillations["decrement_end"].values[0]
    start= bounds_for_oscillations["decrement_start"].values[0]
    # bounds_for_oscillations
    if np.isnan(end):
        end = bounds_for_oscillations["end"].values[0]        
    if np.isnan(start):
        start = bounds_for_oscillations["start"].values[0]
    return start,end, bounds_for_oscillations

def get_signal(date=None, tree=None, measurement=None, df = None, probe="Pt3", timestep = 0.01, start=None, end = None, fixed_by=None):
    """
    Gets signal defined by the probe, start, end and either dataframe df 
    or triplet day,tree, measurement.
    Also interpolates to timestep and shifts the signal to have zero mean value. 
    """
    if df is None:
        df = get_data(date, tree, measurement)
    if pd.isna(start):
        start = 0        
    if pd.isna(end):
        end = np.inf        
    df = df[(df["Time"]>start ) 
            &
            (df["Time"]<end )
            ]
    signal = df[(probe,"Y0")].values
    if fixed_by is not None:
        signal = signal - df[(fixed_by,"Y0")].values
    time = df.index.values
    idx = np.isnan(signal)
    signal = signal[~idx]
    if len(signal) == 0:
        print("Warning: empty signal")
        return None,None
    time = time[~idx]
    signal_, time_ = signal, time
    time = np.arange(time[0], time[-1], timestep)
    signal = np.interp(time, time_, signal_)
    signal = signal - np.mean(signal)
    return time, signal

def find_damping(
        date="2021-03-22",
        csvdir="../csv",
        tree="04",
        measurement="3",
        df=None,
        probe = None, start=None, end=None, T=None, dt = 0.01,
        fixed_by = None,
        method = 'hilbert'):
    s_, e_, r_ = get_limits(date, tree, measurement)
    if (probe is None) or (probe == "auto"):
        probe = r_["probe"].values[0]
    if (probe is None) or (pd.isnull(probe)):
        probe = "Pt3"
    if e_ == -1:
        print("End time is -1, damping skipped.")
        return None
    if start is None:
        start = s_
    if end is None:
        end = e_ 
    if df is None:
        df = get_data(date, tree, measurement)
    time, signal = get_signal(df=df, probe=probe, start=start, end=end, fixed_by=fixed_by)
    if time is None or signal is None:
        print("Time or signal are None, skipped determinantion of damping")
        return None

    if T is None:
        df_f = pd.read_csv("csv/results_fft.csv", index_col=[0,1,2])
        freq = df_f.at[(date,f"BK{tree}",f"M0{measurement}"),"Freq"]
        T = 1/freq    
    else:
        freq = 1/T

    fig, axs = plt.subplots(2,1)
    axs[0].plot(time,signal, label='signál')
    axs[0].plot(time,-signal, label='opačný signál')

    axs[1].plot(time,signal, label='signál')
    axs[1].plot(time,-signal, label='opačný signál')
    axs[1].set(yscale = 'log', ylim = (0.1,None) )
    axs[1].grid()
    
    for ax in axs:
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_color(date2color[date])
            ax.spines[axis].set_linewidth(2)

    if method == 'hilbert':
        amplitude_envelope = np.abs(hilbert(signal))
        k,q = np.polyfit(time[:-1], 
                         np.log(amplitude_envelope[:-1]), 1)
        axs[1].plot(time[:-1], amplitude_envelope[:-1], label='obálka')
    elif method == 'peaks':
        smooth_signal = savgol_filter(signal, 100, 2)
        peaks, _ = find_peaks(np.abs(smooth_signal), distance=50)
        axs[0].plot(time[peaks], signal[peaks], "o", markersize=10)
        k,q = np.polyfit(time[peaks], 
                         np.log(np.abs(signal[peaks])), 1)
    elif method == 'wavelet':
        fig = plt.subplots()
        wavelet = "cmor1-1.5"
        data = signal
        scale = pywt.frequency2scale(wavelet, [freq*dt])
        coef, freqs = pywt.cwt(data, scale, wavelet,
                                sampling_period=dt)
        fig, ax = plt.subplots()
        coef = np.abs(coef)[0,:]
        maximum = np.argmax(coef)
        ax.plot(time, coef, label=freq)
        ax.plot(time[maximum], coef[maximum], "o")
        # ax.set(title="Waveletova transformace signalu pomoci vlnek")
        try:
            k,q = np.polyfit(time[maximum:-maximum], np.log(coef[maximum:-maximum]), 1)
            _ = np.linspace(time[maximum], time[-maximum])
            ax.plot(_,np.exp(k*_+q))
        except:
            k, q = 0, 0
        if len(time)<2*maximum:
            print("Inteval too short for wavelets")
            k, q = 0, 0
        ax.set(yscale='log')
        ax.grid()
        
    t=np.linspace(start, end)
    try:
        axs[0].plot(t,np.exp(k*t+q),t, -np.exp(k*t+q), color='gray')    
        axs[1].plot(time[:-1], np.exp(k*time[:-1]+q), label=f'linearizace, $k={k:.5f}$', color='gray')
    except:
        pass
    fig.suptitle(f"{date}, BK{tree}, M0{measurement}, {probe}, method: {method}", color=date2color[date])
    
    fig2, ax2 = plt.subplots()
    sig = df[(probe,"Y0")]
    t = df.index
    if fixed_by is not None:
        sig = sig - df[(fixed_by,"Y0")]
    sig = sig - sig[0]
    ax2.plot(t,sig)
    idx = (t>start) & (t<end)
    ax2.plot(t[idx], sig[idx], color='red')
    # df_kopie[(probe,"Y0")].plot(ax=ax2)
    # df_kopie.loc[start:,[(probe,"Y0")]].loc[:end,:].plot(ax=ax2, color='red', legend=None)
    ax2.set(title=f"Oscillations {date} BK{tree} M0{measurement}, probe ({probe},Y0)")
    
    return {'figure': fig, 'damping': -k*T, 'figure_fulldomain':fig2, 'signal':signal, 'time':time}


methods = ['hilbert','peaks', 'wavelet']
def main():

    dampings = {}
    for date,tree, measurement in get_all_measurements().values:
        print(f"{date} BK{tree} M0{measurement}")
        dfcsv = get_data(date, tree, measurement)
        ans = [find_damping(
            date=date, tree=tree, measurement=measurement, df=dfcsv, method=method
            ) for method in methods]
        if None in ans:
            dampings[(date,f"BK{tree}", f"M0{measurement}")] = ["None"]*3
            continue
        for i in range(len(ans)):
            ans[i]['figure'].savefig(f"damping_output/damping_{date}_BK{tree}_M0{measurement}_{methods[i]}.png")
        ans[0]['figure_fulldomain'].savefig(f"damping_output/damping_{date}_BK{tree}_M0{measurement}.png")
        for i in range(len(ans)):
            plt.close(ans[i]['figure'])
            plt.close(ans[i]['figure_fulldomain'])
        print([i['damping'] for i in ans])
    
        dampings[(date,f"BK{tree}", f"M0{measurement}")] = [a['damping'] for a in ans]
        plt.close('all')
    
    return dampings


#%%

if __name__ == "__main__":
    csv_ans_file = "damping_output/damping_results.csv"

    os.makedirs("damping_output", exist_ok=True)
    ans = main()
    df = pd.DataFrame(ans).T
    df.columns = methods
    df.index = df.index.rename(["date","tree","measurement"])
    df.to_csv(csv_ans_file)
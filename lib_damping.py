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
# import streamlit as st
# import scipy

from lib_dynatree import get_csv, get_all_measurements

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

def get_signal(date=None, tree=None, measurement=None, df = None, probe="Pt3", timestep = 0.01, start=None, end = None):
    """
    Gets signal defined by the probe, start, end and either dataframe df 
    or triplet day,tree, measurement.
    Also interpolates to timestep and shifts the signal to have zero mean value. 
    """
    if df is None:
        df = get_csv(date, tree, measurement)
    if pd.isna(start):
        start = 0        
    if pd.isna(end):
        end = np.inf        
    df = df[(df["Time"]>start ) 
            &
            (df["Time"]<end )
            ]
    signal = df[(probe,"Y0")].values
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
        probe = None, start=None, end=None,
        method = 'hilbert'):
    s_, e_, r_ = get_limits(date, tree, measurement)
    if probe is None:
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
        df = get_csv(date, tree, measurement)
    time, signal = get_signal(df=df, probe=probe, start=start, end=end)
    if time is None or signal is None:
        print("Time or signal are None, skipped determinantion of damping")
        return None
    
    fig, axs = plt.subplots(2,1)
    axs[0].plot(time,signal, label='signál')
    axs[0].plot(time,-signal, label='opačný signál')

    axs[1].plot(time,signal, label='signál')
    axs[1].plot(time,-signal, label='opačný signál')
    axs[1].set(yscale = 'log', ylim = (0.1,None) )
    axs[1].grid()

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
    t=np.linspace(start, end)
    axs[0].plot(t,np.exp(k*t+q),t, -np.exp(k*t+q), color='gray')    
    axs[1].plot(time[:-1], np.exp(k*time[:-1]+q), label=f'linearizace, $k={k:.5f}$', color='gray')
    fig.suptitle(f"{date}, BK{tree}, M0{measurement}, {probe}, k={k:.4f}")
    
    fig2, ax2 = plt.subplots()
    df_kopie = df - df.iloc[0,:]
    df_kopie = df_kopie[~pd.isna(df_kopie.index)]
    df_kopie[probe].plot(ax=ax2)
    df_kopie.loc[start:,probe].loc[:end,:].plot(ax=ax2, color='red', legend=None)
    ax2.set(title=f"Oscillations {date} BK{tree} M0{measurement}")
    
    return {'figure': fig, 'damping': [k,q], 'figure_fulldomain':fig2}

def main(method='peaks'):
    for date,tree, measurement in get_all_measurements().values:
        print(f"{date} BK{tree} M0{measurement}")
        try:
            ans = find_damping(date=date, tree=tree, measurement=measurement, method=method)
        except:
            print("FAILED")
            continue
        if ans is None:
            continue
        ans['figure'].savefig(f"damping_{method}/damping_{date}_BK{tree}_M0{measurement}.png")
        ans['figure_fulldomain'].savefig(f"damping_{method}/oscillation_{date}_BK{tree}_M0{measurement}.png")
        plt.close(ans['figure'])
        plt.close(ans['figure_fulldomain'])
        print(ans['damping'])

        csv_ans_file = f"damping_{method}/damping_results.csv"
        try:
            df_ans = pd.read_csv(csv_ans_file, index_col=[0,1,2], header=0)
        except:
            df_ans = pd.DataFrame(columns=["date","tree","measurement","k","q"])
            df_ans.to_csv(csv_ans_file, index=None)
            df_ans = pd.read_csv(csv_ans_file, index_col=[0,1,2], header=0)
    
        df_ans.loc[(date,f"BK{tree}", f"M0{measurement}"),:] = ans['damping']
        df_ans.to_csv(csv_ans_file)
        df_ans
        plt.close('all')


# ans = find_damping(date = "2021-03-22",tree="04",measurement="4")  # Pt4
# ans = find_damping(date = "2021-03-22",tree="12",measurement="3")
# 2022-08-16 BK11 M03

#%%
# find_damping(method='peaks')

#%%

# time,signal = get_signal(date='2022-04-05', tree="01", measurement="2",start = 96, end=115)

#%%

# from findpeaks import findpeaks
# fp = findpeaks(method='peakdetect')
# results = fp.fit(signal)
# smooth_signal = savgol_filter(signal, 100, 2)
# peaks, _ = find_peaks(np.abs(smooth_signal), distance=50)
# plt.plot(time[peaks], signal[peaks], "o", markersize=10)
# k,q = np.polyfit(time[peaks], 
#                  np.log(np.abs(signal[peaks])), 1)

# plt.plot(time,signal)
# plt.plot(time, smooth_signal)

# plt.plot(time, np.exp(k*time+q), label='obálka', color='gray')
# plt.plot(time, -np.exp(k*time+q), label='obálka', color='gray')

# df = get_csv("2022-08-16","11","3")

# #%%
# df[("Pt3","Y0")].plot()

# #%%

# find_damping(date="2022-08-16", tree="11", measurement="3")

#%%
# for date,tree, measurement in get_all_measurements().values[:2]:
#     find_damping(date=date, tree=tree, measurement=measurement)
methods = ['hilbert','peaks']
if __name__ == "__main__":
    for method in methods:
        os.makedirs(f"damping_{method}", exist_ok=True)
        main(method=method)
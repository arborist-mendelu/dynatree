#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:24:29 2024

@author: marik
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import matplotlib.pyplot as plt
from scipy.signal import hilbert
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
        probe = None, start=None, end=None):
    
    s_, e_, r_ = get_limits(date, tree, measurement)
    # print(r_)
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
    df_data = get_csv(date, tree, measurement)
    time, signal = get_signal(df=df_data, probe=probe, start=start, end=end)
    if time is None or signal is None:
        print("Time or signal are None, skipped determinantion of damping")
        return None
    
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    k,q = np.polyfit(time[:-1], np.log(amplitude_envelope[:-1]), 1)
    # k_lin, q_lin = k,q
    # [k,q]
    # kq = scipy.optimize.least_squares(lambda x: sum((np.exp(x[0]*time[:-1:10]+x[1])-amplitude_envelope[:-1:10])**2),[k,q], ftol=1e-18, gtol=1e-18, max_nfev=10000, 
    #                        #            jac = lambda x:
    #                        # [sum(2*(np.exp(x[0]*time[:-1]+x[1])-amplitude_envelope[:-1]) * np.exp(x[0]*time[:-1]+x[1]) * time[:-1])
    #                        #     ,
    #                        #     sum(2*(np.exp(x[0]*time[:-1]+x[1])-amplitude_envelope[:-1]) * 
    #                        #         np.exp(x[0]*time[:-1]+x[1]))
    #                        #     ]           
    #                        )
    # kq
    # k,q = kq['x']
    fig, axs = plt.subplots(2,1)
    ax =axs[0]
    ax.plot(time,signal, label='signál')
    ax.plot(time,-signal, label='opačný signál')
    t=np.linspace(start, end)
    ax.plot(t,np.exp(k*t+q),t, -np.exp(k*t+q), color='gray')    
    # ax.plot(t,np.exp(k_lin*t+q_lin),t, -np.exp(k_lin*t+q_lin), color='gray', linestyle="--")    
    ax = axs[1]
    ax.plot(time,signal, label='signál')
    ax.plot(time,-signal, label='opačný signál')
    ax.plot(time[:-1], amplitude_envelope[:-1], label='obálka')
    ax.plot(time[:-1], np.exp(k*time[:-1]+q), label=f'linearizace, $k={k:.5f}$', color='gray')
    # ax.plot(time[:-1], np.exp(k_lin*time[:-1]+q_lin), label=f'linearizace, $k_{{lin}}={k_lin:.5f}$', color='gray', linestyle="--")
    ax.set(yscale = 'log', ylim = (0.1,None) )
    ax.grid()
    fig.suptitle(f"{date}, BK{tree}, M0{measurement}, {probe}, k={k:.4f}")
    
    fig2, ax2 = plt.subplots()
    df_kopie = df_data - df_data.iloc[0,:]
    df_kopie[probe].plot(ax=ax2)
    df_kopie.loc[start:,probe].loc[:end,:].plot(ax=ax2, color='red', legend=None)
    ax2.set(title=f"Oscillations {date} BK{tree} M0{measurement}")
    
    return {'figure': fig, 'damping': [k,q], 'figure_fulldomain':fig2}

def main():
    for date,tree, measurement in get_all_measurements().values:
        print (f"{date} BK{tree} M0{measurement}")
        try:
            ans = find_damping(date=date, tree=tree, measurement=measurement)
        except:
            print("FAILED")
            continue
        if ans is not None:
            ans['figure'].savefig(f"damping/damping_{date}_BK{tree}_M0{measurement}.png")
            ans['figure_fulldomain'].savefig(f"damping/oscillation_{date}_BK{tree}_M0{measurement}.png")
            plt.close(ans['figure'])
            plt.close(ans['figure_fulldomain'])
            print(ans['damping'])

            csv_ans_file = "damping/damping_results.csv"
            try:
                df_ans = pd.read_csv(csv_ans_file, index_col=[0,1,2], header=0)
            except:
                df_ans = pd.DataFrame(columns=["date","tree","measurement","k","q"])
                df_ans.to_csv(csv_ans_file, index=None)
                df_ans = pd.read_csv(csv_ans_file, index_col=[0,1,2], header=0)
        
            df_ans.loc[(date,f"BK{tree}", f"M0{measurement}"),:] = ans['damping']
            df_ans.to_csv(csv_ans_file)
            df_ans


# for date,tree, measurement in get_all_measurements().values[:2]:
#     find_damping(date=date, tree=tree, measurement=measurement)
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 19:31:14 2023

@author: marik
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import streamlit as st
import scipy

from lib_dynatree import read_data, directory2date, date2dirname

csv_ans_file = "damping/damping_results.csv"

@st.cache_data
def get_df(date, tree, measurement):
    print("loading csv file")
    file = f"../{date}/csv/BK{tree}_M0{measurement}.csv"
    file
    return read_data(file)

def get_limits(date, tree, measurement, csvdir="csv"):
    df_remarks = pd.read_csv(f"{csvdir}/oscillation_times_remarks.csv")
    bounds_for_oscillations = df_remarks[(df_remarks["tree"]==f"BK{tree}") & (df_remarks["measurement"]==f"M0{measurement}") & (df_remarks["date"]==directory2date(date))]
    end = bounds_for_oscillations["decrement_end"].values[0]
    start= bounds_for_oscillations["decrement_start"].values[0]
    bounds_for_oscillations
    if np.isnan(end):
        end = bounds_for_oscillations["end"].values[0]        
    if np.isnan(start):
        start = bounds_for_oscillations["start"].values[0]        
    return (start,end, bounds_for_oscillations['remark'].values[0])

def get_signal(date, tree, measurement, probe="Pt3", timestep = 0.01, start=0, end = 1e10):
    df = get_df(date, tree, measurement)
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
    time = time[~idx]
    signal_, time_ = signal, time
    time = np.arange(time[0], time[-1], timestep)
    signal = np.interp(time, time_, signal_)
    signal = signal - np.mean(signal)
    return time, signal

def find_damping(
        date="01_Mereni_Babice_22032021_optika_zpracovani",
        csvdir="../csv",
        tree="04",
        measurement="3",
        probe = "Pt3", start=0, end=1e10):
    time, signal = get_signal(date, tree, measurement, probe,start=start, end=end)
    
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    k,q = np.polyfit(time[:-1], np.log(amplitude_envelope[:-1]), 1)
    k_lin, q_lin = k,q
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
    ax.plot(time,-signal, label='signál')
    t=np.linspace(start, end)
    ax.plot(t,np.exp(k*t+q),t, -np.exp(k*t+q), color='gray')    
    ax.plot(t,np.exp(k_lin*t+q_lin),t, -np.exp(k_lin*t+q_lin), color='gray', linestyle="--")    
    ax = axs[1]
    ax.plot(time,signal, label='signál')
    ax.plot(time,-signal, label='opačný signál')
    ax.plot(time[:-1], amplitude_envelope[:-1], label='obálka')
    ax.plot(time[:-1], np.exp(k*time[:-1]+q), label=f'linearizace, $k={k:.5f}$', color='gray')
    ax.plot(time[:-1], np.exp(k_lin*time[:-1]+q_lin), label=f'linearizace, $k_{{lin}}={k_lin:.5f}$', color='gray', linestyle="--")
    ax.set(yscale = 'log', ylim = (0.1,None) )
    ax.grid()
    fig.suptitle(f"{directory2date(date)}, BK{tree}, M0{measurement}, {probe}, k={k:.4f}")
    return {'figure': fig, 'damping': [k,q]}

        
#%%

"""
## Day, tree, measurement
"""

def split_path(file):
    data = file.split("/")
    return [file,directory2date(data[1]), data[1], data[3][2:4], data[3][7]]

@st.cache_data
def get_all_measurements():
    files = glob.glob("../01_*/csv/BK*.csv")    
    out = [split_path(file) for file in files]
    return out

df = pd.DataFrame([[i[1],i[3],i[4]] for i in get_all_measurements()], columns=['day','tree', 'measurement'])
df = df.sort_values(by=list(df.columns))
df = df.reset_index(drop=True)

cs = st.columns(2)

with cs[0]:
    columns = st.columns(3)
    
    with columns[0]:
        day = st.radio("Day",list(df['day'].unique()))
    
    df_day = df[df['day']==day]
    with columns[1]:
        tree = st.radio("Tree",list(df_day['tree'].unique()), horizontal=True)
    
    df_measurement = df_day[df_day['tree']==tree]
    with columns[2]:
        measurement = st.radio("Measurement",list(df_measurement['measurement'].unique()), horizontal=True)
    
    df_data = get_df(date2dirname(day), tree, measurement)
    
    with columns[2]:
        probe = st.radio("Probe",["Pt3","Pt4"])
    
    start,end, remark = get_limits(date=date2dirname(day), tree=tree, measurement=measurement)
    # st.write(f"Limits for decrement: from {start} to {end}.")
    
    """
    ## Limits
    """
    
    columns = st.columns(3)
    if np.inf == end:
        end = np.nanmax(df_data["Time"].values)
    with columns[0]:
        new_start = st.number_input("From",value=start)
    with columns[1]:
        new_end = st.number_input("To",value=end)
    
    start = new_start
    end = new_end
    
    with columns[2]:
        if st.button('Save'):
            df_times = pd.read_csv("csv/oscillation_times_remarks.csv", index_col=[0,1,2])
            df_times.at[(day,f"BK{tree}",f"M0{measurement}"),"decrement_end"] = end
            df_times.at[(day,f"BK{tree}",f"M0{measurement}"),"decrement_start"] = start
            df_times.to_csv("csv/oscillation_times_remarks.csv")
            st.rerun()
    
    st.write(f"Remark: {remark}")

    sol = find_damping(date=date2dirname(day), 
                   tree=tree, measurement=measurement, probe=probe, start=start, end=end)
    try:
        df_ans = pd.read_csv(csv_ans_file, index_col=[0,1,2], header=0)
    except:
        df_ans = pd.DataFrame(columns=["date","tree","measurement","k","q"])
        df_ans.to_csv(csv_ans_file, index=None)
        df_ans = pd.read_csv(csv_ans_file, index_col=[0,1,2], header=0)

    df_ans.loc[(day,f"BK{tree}", f"M0{measurement}"),:] = sol['damping']
    df_ans.to_csv(csv_ans_file)
    df_ans


with cs[1]:
    sol['figure']
    sol['figure'].savefig(f"damping/damping_{day}_BK{tree}_M0{measurement}.png")
    sol['damping']
    # st.write(df_data[(probe,"Y0")].head())
    # st.write(df_data[(probe,"X0")].head())

    """
    * Blue curve - original signal
    * Orange curve - original signal multiplied by -1
    * Gray curve - envelope from decreasing exponential, nonlinear least squares method
    * Gray dashed curve - envelope from decreasing exponential, linear leas squares method for logaritm of the data
    """
    
    fig, ax = plt.subplots()
    df_kopie = df_data - df_data.iloc[0,:]
    df_kopie[probe].plot(ax=ax)
    df_kopie.loc[start:,probe].loc[:end,:].plot(ax=ax, color='red', legend=None)
    ax.set(title=f"Oscillations {day} BK{tree} M0{measurement}")
    st.pyplot(fig)
    fig.savefig(f"damping/oscillations_{day}_BK{tree}_M0{measurement}.png")




#%%

# for file, _, date,tree, measurement in get_all_measurements():
#     print(directory2date(date),tree,measurement)
#     try:
#         sol = find_damping(date=date, tree=tree, measurement=measurement)
#         sol['figure'].savefig(f"damping/{directory2date(date)}_{tree}_{measurement}.png")
#     except:
#         print("failed")
#         print(f"find_damping(date=\"{date}\", tree={tree}, measurement={measurement})")

#%%

# find_damping(date="01_Mereni_Babice_05042022_optika_zpracovani", tree=21, measurement=3)

#%%


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:37:18 2024

@author: marik
"""

import pandas as pd
import glob
import scipy.io
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import FFT_spectrum as fftdt
import lib_dynatree as dt

fs = 100 # resampled signal
files = glob.glob("../01_Mereni_Babice_*_optika_zpracovani/csv/*")
data = [{'year':i[24:28], 'month': i[22:24], 'day': i[20:22], 'measurement': i[-6:-4], 'tree':i[-10:-8],
         'date': f"{i[24:28]}-{i[22:24]}-{i[20:22]}"} for i in files]

df_osc_times = pd.read_csv("csv/oscillation_times_remarks.csv")

#%%


"""
## Day, tree, measurement
"""
columns = st.columns(3)

dates = np.unique(np.array([f"{i['date']}"for i in data]))
dates.sort()
with columns[0]:
    date = st.radio("Day",dates)

trees = np.unique(np.array([i['tree'] for i in data if i['date']==date]))
trees.sort()
with columns[1]:
    tree = st.radio("Tree",trees, horizontal=True)


measurements = np.array([i['measurement'] for i in data if i['date']==date and i['tree']==tree])
measurements.sort()
with columns[2]:
    measurement = st.radio("Measurement",measurements, horizontal=True)
    acc_axis = st.radio("Axis",["x","y","z"], horizontal=True)


#%%

year,month,day=date.split("-")
df_optics = dt.read_data(f"../01_Mereni_Babice_{day}{month}{year}_optika_zpracovani/csv/BK{tree}_M{measurement}.csv")
df_acc = pd.read_csv(f"../acc/csv/{date}-BK{tree}-M{measurement}.csv", index_col=0)

#%%

t = np.arange(0, len(df_acc))/100
df_acc.index = t

#%%

row = df_osc_times[
    (df_osc_times['date']==f"{date}") &
    (df_osc_times['tree']==f"BK{tree}") &
    (df_osc_times['measurement']==f"M{measurement}")
     ]
start, end = row.loc[:,["start","end"]].values.reshape(-1)
probe = row['probe'].iat[0]
if pd.isna(probe):
    probe = "Pt3"
acc_columns = [i for i in df_acc.columns if acc_axis in i or acc_axis.upper() in i]
df = df_acc[acc_columns].abs().idxmax()

release_acc = df[df.sub(df.mean()).div(df.std()).abs().lt(1)].mean()
release_optics = dt.find_release_time_optics(df_optics)
df_acc.index = df_acc.index - release_acc + release_optics

columns = st.columns(3)

for i,limits in enumerate([[0, max(df_acc.index[-1],df_optics.index[-1])], [release_optics-1,release_optics+1], [start,end]]): 
    fig, ax = plt.subplots(figsize=(10,15))
    df_acc.loc[limits[0]:limits[1],acc_columns].plot(ax=ax,subplots=True)
    # df_optics[(probe,"Y0")].plot()
    # ax[0].axvline(release_optics, color='k')
    # ax[1].axvline(release_optics, color='k')

    # for a in ax:
    #     a.set(xlim=limits, ylim=(None, None))
    with columns[i]:
        st.pyplot(fig)
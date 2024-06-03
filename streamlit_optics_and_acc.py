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

plt.rcParams["figure.figsize"] = (10,4)

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
df_optics = df_optics - df_optics.iloc[0,:]
df_acc = pd.read_csv(f"../acc/csv/{date}-BK{tree}-M{measurement}.csv", index_col=0)

#%%

"Probes from optics"

options_data = [i for i in df_optics.columns if "Y" in i[1]]
options_data.sort()
options = st.multiselect(
    "Which probes from Optics you want to plot and analyze?",
    options_data,
    [("Pt3","Y0")])

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
f"Start: {start}, End: {end}"

start = st.number_input("start", value=start)
probe = row['probe'].iat[0]
if pd.isna(probe):
    probe = "Pt3"
acc_columns = [i for i in df_acc.columns if "_"+acc_axis in i or "_"+acc_axis.upper() in i]
df = df_acc[acc_columns].abs().idxmax()

release_acc = df[df.sub(df.mean()).div(df.std()).abs().lt(1)].mean()
release_optics = dt.find_release_time_optics(df_optics)
df_acc.index = df_acc.index - release_acc + release_optics

f"""
* Release time is {release_optics} in optics time. 
* Release time in ACC time has been establieshed from the ACC in the folloiwing table. 
* The difference between both realease times is {release_acc-release_optics}.
"""
df[df.sub(df.mean()).div(df.std()).abs().lt(1)].T

columns = st.columns(4)

for c,text in zip(columns, ["Full time domain","Release ±1 sec", "Oscillations", "FFT analysis"]):
    with c:
        "## "+text

def create_images(df, column=None, start=None, end=None, release_optics=None, tmax = 1e10):
    ans = []
    for i,limits in enumerate([[0, tmax], [release_optics-1,release_optics+1], [start,end]]): 
        fig, ax = plt.subplots()
        df.loc[limits[0]:limits[1],column].plot(ax=ax,lw=2)
        ax.grid()
        if i < 2:
            # df_optics[(probe,"Y0")].plot()
            ax.axvline(release_optics, color='k', alpha=0.4, lw=0.5, linestyle="--")
            # ax[1].axvline(release_optics, color='k')
        ans += [fig]
    out = fftdt.do_fft_for_one_column(df.loc[start:end, :], column, create_image=False)
    fig = fftdt.create_fft_image(**out, only_fft=True, ymin = 0.0001)
    ans += [fig]
    return ans, out

for i,c in enumerate(acc_columns):
    ans, out = create_images(df=df_acc, column=c, start=start, end=end, release_optics=release_optics)
    f"### {c}, freq = {out['peak_position']:.3f} Hz"
    columns = st.columns(4)
    for j,a in enumerate(ans):
        with columns[j]:
            st.pyplot(a)

df_optics = fftdt.interp(df_optics, np.arange(df_optics.index[0],df_optics.index[-1],0.01))

for i,c in enumerate(options):
    ans, out = create_images(df=df_optics, column=c, start=start, end=end, release_optics=release_optics)
    f"### {c}, freq = {out['peak_position']:.3f} Hz"
    columns = st.columns(4)
    for j,a in enumerate(ans):
        with columns[j]:
            st.pyplot(a)

        
# out = fftdt.do_fft_for_one_column(df_acc.loc[start:end, :], acc_columns[0], create_image=False)
# fig = fftdt.create_fft_image(**out)

# with columns[-1]:        
#     st.pyplot(fig)
        
# for option in options:        
#     pass
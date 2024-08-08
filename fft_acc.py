#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:37:18 2024

Cte csv soubory z optiky a z konverze Matlab->csv pro akcelerometry a pocita fft
a data uklada do podadresare temp_figs.

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
import pickle 


def create_images(df, column=None, start=None, end=None, release_optics=None, tmax = 1e10, date="", tree="", measurement=""):
    ans = []
    if isinstance(column, tuple):
        name = column[0]
    else:
        name = column
    print(f"probe {name} from {start} to {end}")
    for i,limits in enumerate([[0, tmax], [release_optics-1,release_optics+1], [start,end]]): 
        fig, ax = plt.subplots()
        plt.gca().set(title=f"{date} BK{tree} M{measurement} {column}")
        plt.tight_layout()
        df.loc[limits[0]:limits[1],column].plot(ax=ax,lw=2)
        ax.grid()
        if i < 2:
            # df_optics[(probe,"Y0")].plot()
            ax.axvline(release_optics, color='k', alpha=0.4, lw=0.5, linestyle="--")
            # ax[1].axvline(release_optics, color='k')
        ans += [fig]
        plt.savefig(f"temp_figs/{date}_BK{tree}_M{measurement}_{name}_{i}.png")
    out = fftdt.do_fft_for_one_column(df.loc[start:end, :], column, create_image=False)
    if out is not None:
        fig = fftdt.create_fft_image(**out, only_fft=True, ymin = 0.0001)
        plt.gca().set(title=f"{date} BK{tree} M{measurement} {column}")
        plt.tight_layout()
        plt.savefig(f"temp_figs/{date}_BK{tree}_M{measurement}_{name}_{i+1}.png")
        ans += [fig]
    plt.close('all')
    return None, out

#%%
plt.rcParams["figure.figsize"] = (8,5)

fs = 100 # resampled signal
files = glob.glob("../acc/csv/*")
# files = pd.read_csv("temp_figs/failed.txt").values.reshape(-1)
#%%
data = [{'year':i[11:15], 'month': i[16:18], 'day': i[19:21], 'measurement': i[-6:-4], 'tree':i[-10:-8],
         'date': f"{i[11:15]}-{i[16:18]}-{i[19:21]}", 'file':i} for i in files]
data

#%%
df_osc_times = pd.read_csv("csv/oscillation_times_remarks.csv", index_col=[0,1,2])


#%%

def zprocesuj(dataset):
    file = dataset['file']
    print(file)
    df_acc = pd.read_csv(file) 
    t = np.arange(0, len(df_acc))/100
    df_acc.index = t
    
    date,tree, measurement = dataset['date'], dataset['tree'],dataset['measurement']
    day, month, year = dataset['day'], dataset['month'], dataset['year']
    
    df_optics = dt.read_data(f"../01_Mereni_Babice_{day}{month}{year}_optika_zpracovani/csv/BK{tree}_M{measurement}.csv")
    df_optics = df_optics - df_optics.iloc[0,:]
    
    
    row = df_osc_times.loc[(date,f"BK{tree}",f"M{measurement}"),:]
    start, end = row['start'], row['end']
    
    acc_axis = "z"
    acc_columns = [i for i in df_acc.columns if "_"+acc_axis in i or "_"+acc_axis.upper() in i]
    df = df_acc[acc_columns].abs().idxmax()
    
    release_acc = df[df.sub(df.mean()).div(df.std()).abs().lt(1)].mean()
    release_optics = dt.find_release_time_optics(df_optics)
    df_acc.index = df_acc.index - release_acc + release_optics
    
    # print(f"Release time is {release_optics}. It has been establieshed from the following ACC.\n", df[df.sub(df.mean()).div(df.std()).abs().lt(1)].T)
    
    coords = {'date': date, 'tree': tree, 'measurement': measurement}
    ans = {}
    out = {}
    for i,c in enumerate(acc_columns):
        ans[c], out[c] = create_images(df=df_acc, column=c, start=start, end=end, release_optics=release_optics, **coords)
        # f"### {c}, freq = {out['peak_position']:.3f} Hz"
        # columns = st.columns(4)
        # for j,a in enumerate(ans):
        #     with columns[j]:
        #         st.pyplot(a)
    
    df_optics = fftdt.interp(df_optics, np.arange(df_optics.index[0],df_optics.index[-1],0.01))
    
    for i,c in enumerate([("Pt3","Y0"), ("Pt4","Y0")]):
        ans[c], out[c] = create_images(df=df_optics, column=c, start=start, end=end, release_optics=release_optics, **coords)
        # f"### {c}, freq = {out['peak_position']:.3f} Hz"
        # columns = st.columns(4)
        # for j,a in enumerate(ans):
        #     with columns[j]:
        #         st.pyplot(a)

    df_elasto = dt.read_data(f"../01_Mereni_Babice_{day}{month}{year}_optika_zpracovani/csv_extended/BK{tree}_M{measurement}.csv")
    df_elasto = df_elasto - df_elasto.iloc[0,:]
    c = ('Elasto(90)', 'nan')
    # df_elasto = df_elasto[[c]]

    ans[c], out[c] = create_images(df=df_elasto, column=c, start=start, end=end, release_optics=release_optics, **coords)
        # f"### {c}, freq = {out['peak_position']:.3f} Hz"
        # columns = st.columns(4)
        # for j,a in enumerate(ans):
        #     with columns[j]:
        #         st.pyplot(a)
    
    out['data']=coords
    with open(f'temp_figs/data_{date}_BK{tree}_M{measurement}.pkl', 'wb') as f:
        pickle.dump(out, f)
    plt.close('all')

# %%
try:
    success = pd.read_csv("temp_figs/success.txt").values.reshape(-1)
except:
    success = []
    
for dataset in data:
    if dataset['file'] in success:
        print(f"Skipping {dataset['file']}")
        continue
    try:
        zprocesuj(dataset)
        with open('temp_figs/success.txt', 'a') as f:
            f.write(dataset['file']+"\n")
    except:
        print("FAILED")
        with open('temp_figs/failed.txt', 'a') as f:
            f.write(dataset['file']+"\n")
        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:15:16 2023

@author: marik
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy 
import sys
import os
from lib_dynatree import read_data, get_chains_of_bendlines
from scipy import interpolate
from scipy.fft import fft, fftfreq

cesta = "../01_Mereni_Babice_22032021_optika_zpracovani"
name =  "BK21_M03"

df = read_data(f"{cesta}/csv/{name}.csv")

# Bendlines in the middle, on the left and on the right
BL_chain_Y = get_chains_of_bendlines()
BL_chain_X = get_chains_of_bendlines(axis="X")

# Keep only the data on bendlines in memory
df = df[sum(BL_chain_X, start=[])+sum(BL_chain_Y,start=[])]

# %%

fig, ax = plt.subplots()

df_delta = df - df.iloc[0,:]   # Pracuje se se změnou oproti nulovému času 
df_delta.plot(y=[i for i in BL_chain_Y[0] if "BL44" in i[0]]) # Vykresli všechno na B44

# %%

fig, ax = plt.subplots()

idx = [(f"BL{i}","Pt0BX") for i in [51,59,67]]
idy = [(f"BL{i}","Pt0BY") for i in [51,59,67]]
x0 = df.loc[0,idx]
y0 = df.loc[0,idy]
for i in [0,1,2]:
    y = df.loc[0,BL_chain_Y[i]]-y0.loc[idy[0]]
    x = df.loc[0,BL_chain_X[i]]-x0.loc[idx[0]]
    plt.plot(y,x,"o", color=f"C{i}")

# %%

fig, ax = plt.subplots()

release_time = df.loc[:,("BL44","Pt0AY")].idxmin()
time=release_time
time = [i for i in df.index if i>release_time+2][0]

for i in range(3):
    y = 10*df_delta.loc[time,BL_chain_Y[i]] + y0.loc[BL_chain_Y[i][-1]] - y0.loc[BL_chain_Y[0][-1]]
    x = df.loc[time,BL_chain_X[i]]-x0.loc[BL_chain_X[i][-1]]
    plt.plot(y,x,"-", color=f"C{i}")


# # %%
# df2.columns[0]
# # %%

# fs = 100
# fig,ax = plt.subplots()
# for i in df2.columns[::7]:
#     signal = df2.loc[:,i]
#     signal = signal - signal.mean()
#     time = df2.index
#     f = interpolate.interp1d(time, signal)
#     time_fft = np.arange(time[0],time[-1],1/fs)
#     N = time_fft.shape[0]
#     signal_fft = f(time_fft)
    
#     yf = fft(signal_fft)  # preform FFT analysis
#     xf_r = fftfreq(N, 1/fs)[:N//2]
#     yf_r = 2.0/N * np.abs(yf[0:N//2])
#     peak_index = np.argmax(yf_r[2:])+2  # find the peak, exclude the start
#     peak_position = xf_r[peak_index]
#     delta_f = np.diff(xf_r).mean()
    
#     ax.plot(xf_r, yf_r, color='gray', alpha=0.1)
#     ax.plot(xf_r[peak_index],yf_r[peak_index],"o", color='red')

# ax.set(xlim=(0,10))
# ax.grid()
# ax.set(ylabel="FFT", xlabel="Freq./Hz", yscale='log', ylim=(0.0001,None))



# # %%

# for b in BLines[0]:
#     fig, ax = plt.subplots()
#     df2[b].plot(ax = ax)
#     ax.set(title=b)

# # %%

# fig, ax = plt.subplots()
# df2[["BL44"]].plot(ax = ax, color="C0")
# df2[["BL49"]].plot(ax = ax, color="C1")



# # %%

# column = df2.columns
# signal = df2[column]    
# df2i = df2.interpolate(axis=0, method='nearest')

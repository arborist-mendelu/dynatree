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
from lib_dynatree import read_data
from scipy import interpolate
from scipy.fft import fft, fftfreq


# Bendlines in the middle, on the left and on the right
BLines = np.array([[f"BL{i}" for i in range(j,j+8)] for j in [44,52,60] ])

cesta = "01_Mereni_Babice_22032021_optika_zpracovani"
name =  "BK21_M03"

df = read_data(f"{cesta}/csv/{name}.csv")

# %%
BLines_mask={}
for coord in ["X","Y"]:
    BLines_mask[coord]=[[],[],[]]
    for i_ in range(len(BLines)):
        B = BLines[i_]
        temp_long = [f"Pt0A{coord}", *[f"{coord}{i}" for i in range(10)], f"Pt0B{coord}"]
        temp_short = [f"Pt0A{coord}", *[f"{coord}{i}" for i in range(5)], f"Pt0B{coord}"]
        temp = [temp_short,*[temp_long for i in range(8)]]
        BLines_mask_ = [ [(j,i_) for i_ in i] for j,i in zip (B,temp)]
        BLines_mask_ = [j for i in BLines_mask_ for j in i]
        BLines_mask[coord][i_] = BLines_mask_
        
df2 = df.loc[72:,BLines_mask["Y"][0]]
df2x = df.loc[72:,BLines_mask["X"][0]]

for i in [0,1,2]:
    y = df[BLines_mask["Y"][i]].iloc[0,:]
    x = df[BLines_mask["X"][i]].iloc[0,:]
    plt.plot(y,x,"o", color=f"C{i}")

# %%
df2.columns[0]
# %%

fs = 100
fig,ax = plt.subplots()
for i in df2.columns[::7]:
    signal = df2.loc[:,i]
    signal = signal - signal.mean()
    time = df2.index
    f = interpolate.interp1d(time, signal)
    time_fft = np.arange(time[0],time[-1],1/fs)
    N = time_fft.shape[0]
    signal_fft = f(time_fft)
    
    yf = fft(signal_fft)  # preform FFT analysis
    xf_r = fftfreq(N, 1/fs)[:N//2]
    yf_r = 2.0/N * np.abs(yf[0:N//2])
    peak_index = np.argmax(yf_r[2:])+2  # find the peak, exclude the start
    peak_position = xf_r[peak_index]
    delta_f = np.diff(xf_r).mean()
    
    ax.plot(xf_r, yf_r, color='gray', alpha=0.1)
    ax.plot(xf_r[peak_index],yf_r[peak_index],"o", color='red')

ax.set(xlim=(0,10))
ax.grid()
ax.set(ylabel="FFT", xlabel="Freq./Hz", yscale='log', ylim=(0.0001,None))



# %%

for b in BLines[0]:
    fig, ax = plt.subplots()
    df2[b].plot(ax = ax)
    ax.set(title=b)

# %%

fig, ax = plt.subplots()
df2[["BL44"]].plot(ax = ax, color="C0")
df2[["BL49"]].plot(ax = ax, color="C1")



# %%

column = df2.columns
signal = df2[column]    
df2i = df2.interpolate(axis=0, method='nearest')

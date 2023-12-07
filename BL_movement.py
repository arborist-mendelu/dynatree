#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:15:16 2023

@author: marik
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from lib_dynatree import read_data, get_chains_of_bendlines, find_release_time_optics
from scipy import interpolate
from scipy.fft import rfft, rfftfreq
import shutil


DATE, TREE, MEASUREMENT = "2021-03-22", "BK21", "M03"
MEASUREMENT = "M02"

DATE, TREE, MEASUREMENT = "2022-04-05", "BK12", "M04"
DATE, TREE, MEASUREMENT = "2021-03-22", "BK08", "M05"

cesta = f"../01_Mereni_Babice_{''.join(reversed(DATE.split('-')))}_optika_zpracovani"
cesta

cam = 1

df = read_data(f"{cesta}/csv/{TREE}_{MEASUREMENT}.csv")

# Bendlines in the middle, on the left and on the right
BL_chain_Y = get_chains_of_bendlines(cam=cam)
BL_chain_X = get_chains_of_bendlines(axis="X", cam=cam)

# Keep only the data on bendlines in memory
df = df[sum(BL_chain_X, start=[])+sum(BL_chain_Y,start=[])]
df_delta = df - df.iloc[0,:]   # Pracuje se se změnou oproti nulovému času 


if cam == 0:
    bottom_bls = [17, 25, 33]
    top_middle_bl = 10
else:
    bottom_bls = [51, 59, 67]
    top_middle_bl = 44

# %%
# start = 68
# end=1000

# fig, ax = plt.subplots()

# df_delta.loc[start:end,[i for i in BL_chain_Y[0] if f"BL{top_middle_bl}" in i[0]]].plot(ax=ax) # Vykresli všechno na B44 nebo B10, podle kamery

# %%

fig, ax = plt.subplots()

origin = {i: df.at[0,("BL51",f"Pt0B{i}")] for i in ["X","Y"]}
for i in [0,1,2]:
    y = df.loc[0,BL_chain_Y[i]]-origin["Y"]
    x = df.loc[0,BL_chain_X[i]]-origin["X"]
    plt.plot(y,x,".", color=f"C{i}")

# %%

# TEMP_DIR = 'temp/movement'
# shutil.rmtree(TEMP_DIR, ignore_errors=True)
# os.makedirs(TEMP_DIR, exist_ok=True)  

release_time = np.abs(df_delta.loc[:,(f"BL{top_middle_bl}","Pt0AY")]).idxmax()

df_interpolated = pd.DataFrame(index=np.arange(release_time, df.index[-1],0.01), columns=df.columns)
for col in df.columns:
    f = interpolate.interp1d(df.index, df[col])
    df_interpolated[col] = f(df_interpolated.index)

df_interpolated_delta = df_interpolated - df.iloc[0,:]   # Pracuje se se změnou oproti nulovému času 

# %%

fig,ax = plt.subplots()

figs = []
N = len(df_interpolated.index)

for n,time in enumerate(df_interpolated.index):
    print(f"\r{100*n/N:.02f} percent finished",end="\r",flush=True)

    fig,ax = plt.subplots()
    for i in range(3):
        y = df_interpolated_delta.loc[time,BL_chain_Y[i]]
        x = df_interpolated.loc[time,BL_chain_X[i]] - origin['X']
        ax.plot(y,x,"-", color=f"C{i}")
    ax.set(
           xlim=(-5,10), 
           ylim=(0,4000),
           ylabel="vertical position / mm", 
           xlabel="horizontal displacement / mm",
           title=f"Time {time:.3f}"
           )
    # plt.savefig(f"{TEMP_DIR}/{DATE}_{TREE}_{MEASUREMENT}_{n:08}.png")
    figs = figs + [fig]
    plt.close(fig)
    # break

# %%


# # %%
# command = f"ffmpeg -y -framerate 10 -i {TEMP_DIR}/{DATE}_{TREE}_{MEASUREMENT}_%08d.png -c:v libx264 -r 30 -pix_fmt yuv420p {DATE}_{TREE}_{MEASUREMENT}.mp4"
# os.system(command)

# # %%

# fs = 100
# signal = df_delta.loc[release_time+1:,("BL44","Pt0AY")]
# N = signal.shape[0]
# yf = rfft(signal.values)  # preform FFT analysis
# xf = rfftfreq(N, 1/fs)
# yf_r =  2.0/N * np.abs(yf)
# fig, ax = plt.subplots()
    
# ax.plot(xf, yf_r, '.', color='gray')
# ax.set(xlim=(0,15), yscale='log')
# #     ax.plot(xf_r[peak_index],yf_r[peak_index],"o", color='red')

# %%

# from scipy import signal
# sig = df_interpolated.loc[:,("BL44","Y0")]
# sos = signal.butter(4, 20, 'highpass', fs=100, output='sos')

# # %%
# # b, a = signal.butter(4, 1000, 'high', analog=True)
# b,a = signal.butter(3, 1, 'highpass', fs=100, output='ba')
# w, h = signal.freqs(b, a)

# plt.semilogx(w, 20 * np.log10(abs(h)))
# plt.title('Butterworth filter frequency response')
# plt.xlabel('Frequency [radians / second]')
# # plt.margins(0, 0.1)
# plt.grid(which='both', axis='both')

# # %%

# b, a = signal.butter(4, 100, 'high')

# w, h = signal.freqs(b, a)

# plt.semilogx(w, 20 * np.log10(abs(h)))

# plt.title('Butterworth filter frequency response')

# plt.xlabel('Frequency [radians / second]')

# plt.ylabel('Amplitude [dB]')

# plt.margins(0, 0.1)

# plt.grid(which='both', axis='both')

# plt.axvline(100, color='green') # cutoff frequency

# plt.show()
# # %%

# posun = sig.mean()
# sig = sig - posun

# fig, ax = plt.subplots()
# filtered = signal.sosfilt(sos, sig)

# filtered = filtered + posun

# ax.plot(sig.values)
# ax.plot(filtered)
# ax.set_title('After filter')

# ax.set_xlabel('Time [seconds]')

# plt.tight_layout()

# plt.show()


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

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
from lib_dynatree import read_data, get_chains_of_bendlines, find_release_time_optics, read_data_selected
from lib_dynatree import find_release_time_optics
from scipy import interpolate
from scipy.fft import rfft, rfftfreq, irfft
import shutil
import scipy.signal as signal
import matplotlib.animation as animation

DATE, TREE, MEASUREMENT = "2021-03-22", "BK21", "M03"
MEASUREMENT = "M02"
DATE, TREE, MEASUREMENT = "2022-04-05", "BK12", "M04"
DATE, TREE, MEASUREMENT = "2021-03-22", "BK08", "M05"

cesta = f"../01_Mereni_Babice_{''.join(reversed(DATE.split('-')))}_optika_zpracovani"

cam = 1

# Bendlines in the middle, on the left and on the right
BL_chain_Y = get_chains_of_bendlines(cam=cam)
BL_chain_X = get_chains_of_bendlines(axis="X", cam=cam)

df = read_data_selected(f"{cesta}/csv/{TREE}_{MEASUREMENT}.csv", probes = ["Time"] + [f"BL{i}" for i in range(44,68)])
# %%

# https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
def align_yaxis(ax1, ax2):
    y_lims = np.array([ax.get_ylim() for ax in [ax1, ax2]])

    # force 0 to appear on both axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize both axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lim1, new_lim2 = y_new_lims_normalized * y_mags
    ax1.set_ylim(new_lim1)
    ax2.set_ylim(new_lim2)

def legend_unique(ax=plt.gca()):
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

# Define class for simple manipulation

class Bendlines:
    """
    Class to work with probes on bendlines.
    
    Argument is a dataframe with the data.
    
    Attrributes
    
    * columns_X : column names with X coordiante (along the tree)
    * columns_Y : column names with Y coordinate (the pulling direction)
    """

    def __init__(self, df):
        columns_Y = [i for i in df.columns if "Y" in i[1]]
        columns_X = [i for i in df.columns if "X" in i[1]]
        self.columns_X = columns_X
        self.columns_Y = columns_Y
        self.data = df
        self.delta = df.copy()
        self.delta[columns_Y] = self.delta[columns_Y] -  self.delta.loc[0,columns_Y]
        self.release_time = find_release_time_optics(df, probe="BL44")
        self.time = df.index
        self.is_resampled = False
        self.BL = {}
        for n,i in enumerate(['center','left','right']):
            self.BL[i] = df[[*BL_chain_X[n] ,*BL_chain_Y[n] ]]

    def __repr__(self):
        output = f"""Data on bendlines
Release time: {self.release_time}
Signal length: {self.data.index.max()}
Is resampled: {'time_resampled' in dir(self)}
"""
        return output

    def resample(self, rate=100):
        self.is_resampled = True
        df = self.data
        self.timestep = np.round(1/rate,2)
        time_resampled = np.round(np.arange(0, df.index.max(), self.timestep),2)
        self.time_resampled = time_resampled
        df_resampled = pd.DataFrame(index=self.time_resampled, columns=df.columns)
        for col in df.columns:
            f = interpolate.interp1d(df.index, df[col])
            df_resampled[col] = f(time_resampled)
        self.data_resampled = df_resampled    

        self.delta_resampled = df_resampled.copy()
        self.delta_resampled[self.columns_Y] = self.delta_resampled[self.columns_Y] -  self.delta_resampled.loc[0,self.columns_Y]

        self.BL_resampled = {}
        for n,i in enumerate(['center','left','right']):
            self.BL_resampled[i] = df_resampled[[*BL_chain_X[n] ,*BL_chain_Y[n] ]]
            
    def fft(self, start=None, end=None, padding=2, select_probe=lambda x: "Y" in x[1]):
        if start is None:
            start = self.release_time + padding
        if end is None:
            end = self.time[-1] - padding
        self.fft = {}
        for col in self.data.columns:
            if not select_probe(col):
                continue
            self.fft[col] = {}
            data = self.data_resampled.loc[start:end,col].values
            data = data - data.mean()
            fft_output = rfft(data)
            self.fft[col]['data'] = data
            self.fft[col]['time'] = self.data_resampled.loc[start:end,col].index
            self.fft[col]['fourier'] = fft_output
            self.fft[col]['frequencies'] = rfftfreq(len(data), d = self.timestep)
            self.fft[col]['amplitudes'] = 2*np.abs(fft_output)/len(data)
        pass
            
    def plot(self, list_of_probes=None, time = 0, delta = False, shift=(0,0), **kwds):
        xcoords = [i for i in self.data.columns if i in self.columns_X and i in list_of_probes]
        ycoords = [(i[0], i[1].replace("X","Y")) for i in xcoords]
        if delta:
            df = self.delta
            shift=(0,0)
        else:
            df = self.data
            shift = [df.at[time,i] for i in shift]
        plt.plot(df.loc[time,ycoords]-shift[0], df.loc[time,xcoords]-shift[1],".", **kwds)

test = Bendlines(df)
test.resample()
test.fft(start=0, padding=0)

# %%

times = [0,test.release_time]
for n,time in enumerate(times):
    for i in ['center', 'left','right']:
        test.plot(test.BL[i], delta=False, time=time, shift=(("BL51","Pt0BY"),("BL51","Pt0BX")), color=f"C{n}", label=f"t = {time:.2f}s")
legend_unique()


# %%

fft_data = test.fft[("BL44","Pt0AY")]


fig, ax = plt.subplots(3, 1, figsize=(8, 6))

ax[0].semilogy(fft_data['frequencies'], fft_data['amplitudes'], label="FFT spectrum of the signal")
ax[0].set(ylim=(0.0001,None), xlim=(0,5))


ax[0].grid()

# # Find peaks
# peak_data = signal.find_peaks(fft_data['amplitudes'])
# # Select peaks around 1Hz
# selected_peaks = peak_data[0][np.abs(fft_data['frequencies'][peak_data[0]]-1)<0.5]
# # Find the most prominent peak near 1Hz
# npeak = signal.peak_prominences(fft_data['amplitudes'], selected_peaks)[0].argmax()
# # Find the frequency of the most prominent peak
# freq_second = fft_data['frequ
                       
freq_second = 1.067299140231248                       

# Design the peak filter
fs = 100.0  # Sample frequency (Hz)
Q = 50.0  # Quality factor
b, a = signal.iirpeak(freq_second, Q, fs)

# Frequency response
freq, h = signal.freqz(b, a, fs=fs)
# Plot
ax[1].plot(freq, 20*np.log10(np.maximum(abs(h), 1e-5)), label='Filter, frequency response', color="C1")
ax[1].set_ylabel("Amplitude (dB)", color='C1')
ax[1].set_xlim([0, 5])
ax[1].set_ylim([-50, 10])
ax[1].grid(True)
ax[2].plot(freq, np.unwrap(np.angle(h))*180/np.pi, color='C2', label='Filter, angle')
ax[2].set_ylabel("Angle (degrees)", color='C2')
ax[2].set_xlabel("Frequency (Hz)")
ax[2].set_xlim([0, 5])
ax[2].set_yticks([-90, -60, -30, 0, 30, 60, 90])
ax[2].set_ylim([-90, 90])
ax[2].grid(True)
plt.tight_layout()
fig.legend()
plt.show()

# %%

fft_data = test.fft[("BL44","Pt0AY")]

filtered_fft = fft_data['fourier'].copy()
filtered_fft[:85] = 0 + 0j
filtered_fft[92:] = 0 + 0j

fig, ax = plt.subplots()
ax.semilogy(fft_data['frequencies'],np.abs(fft_data['fourier']))
ax.semilogy(fft_data['frequencies'],np.abs(filtered_fft))
ax.set(xlim=(0,5), ylim=(1,None))

filtered_signal = irfft(filtered_fft)

fig, ax = plt.subplots()
ax.plot(fft_data['time'], fft_data['data'])
ax2=ax.twinx()
# fig, ax = plt.subplots()
ax2.plot(fft_data['time'], filtered_signal, color="C1")
align_yaxis(ax, ax2)


# %%

# from scipy import signal
# b,a = signal.iirfilter(10, [0.75, 1.25], rs=60, btype='bandstop',
#                        analog=False, ftype='butter', fs=1000,
#                        output='ba')
data = test.delta_resampled.loc[:,("BL44","Pt0AY")].values
time = test.time_resampled
data_filtered = signal.filtfilt(b, a, data)

fig, ax = plt.subplots()
ax.plot(time, data)
ax2 = ax.twinx()
ax2.plot(time,data_filtered, color="C1")


align_yaxis(ax, ax2)

# %%
# from scipy.fft import rfft, rfftfreq

fig, ax = plt.subplots()

for data_ in [data, data_filtered]:
    ax.semilogy(rfftfreq(len(data_), d=0.01), np.abs(rfft(data_)))
    ax.set(xlim=(0,5))
    ax.grid()
ax.set(title="Signal and filtered signal")

# %%

probe = ("BL44","Pt0AY")
for d in [test.data,test.data_resampled]:
    d.loc[test.release_time+2:test.time[-1]-2,probe].plot()
    print(d.shape)

# %%

# # Keep only the data on bendlines in memory
# # df = df[sum(BL_chain_X, start=[])+sum(BL_chain_Y,start=[])]
# df = df.loc[:,sum([BL_chain_X[i]+BL_chain_Y[i] for i in [0,1,2]],start=[])]
# df_delta = df - df.iloc[0,:]   # Pracuje se se změnou oproti nulovému času


# if cam == 0:
#     bottom_bls = [17, 25, 33]
#     top_middle_bl = 10
# else:
#     bottom_bls = [51, 59, 67]
#     top_middle_bl = 44

# %%
# start = 68
# end=1000

# fig, ax = plt.subplots()

# df_delta.loc[start:end,[i for i in BL_chain_Y[0] if f"BL{top_middle_bl}" in i[0]]].plot(ax=ax) # Vykresli všechno na B44 nebo B10, podle kamery

# %%

fig, ax = plt.subplots()

origin = {i: test.data.at[0,("BL51",f"Pt0B{i}")] for i in ["X","Y"]}
for i in [0,1,2]:
    y = test.data.loc[0,BL_chain_Y[i]]-origin["Y"]
    x = test.data.loc[0,BL_chain_X[i]]-origin["X"]
    plt.plot(y,x,".", color=f"C{i}")
    
# %%
fig,ax = plt.subplots()

timestep = 2
figs = []
N = test.time_resampled.size
TEMP_DIR = "temp/movement"


for n,time in enumerate(test.time_resampled):
    # if n%100 !=0:
    #     continue
    if time < test.release_time:
        continue
    print(f"\r{100*n/N:.02f} percent finished",end="\r",flush=True)

    fig,ax = plt.subplots()
    df_resampled = test.delta_resampled
    for i in [0,1,2]:
        y = df_resampled.loc[time,BL_chain_Y[i]] #- origin['Y']
        x = df_resampled.loc[time,BL_chain_X[i]] - origin['X']
        ax.plot(y,x,"-", color=f"C{i}")
    ax.set(
            xlim=(-5,10), 
           ylim=(0,4000),
           ylabel="vertical position / mm", 
           xlabel="horizontal displacement / mm",
           title=f"Time {time:.3f}"
           )
    plt.savefig(f"{TEMP_DIR}/{DATE}_{TREE}_{MEASUREMENT}_{n:08}.png")
    figs = figs + [fig]
    plt.close(fig)
    # break
# figs[1]
# %%

# figs[47]
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

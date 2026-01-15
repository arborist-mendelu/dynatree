#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 10:00:36 2024

@author: marik
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from dynatree.dynatree import read_data

import pywt
wavelet = 'cmor1-1.5'

df_remarks = pd.read_csv("csv/oscillation_times_remarks.csv")
df = read_data(f"../01_Mereni_Babice_22032021_optika_zpracovani/csv/BK04_M03.csv")
df = df[df["Time"]>42]


#%%

df[("Pt3","Y0")].plot()
time_ = df["Time"].values
data_ = df[("Pt3","Y0")].values

time = np.arange(time_[0], time_[-1], 0.01)
data = np.interp(time,time_, data_)
data = data - data.mean()

plt.subplots()
plt.plot(time,data)
plt.title("Zkoumana data")

#%%  wavelet
freq = 0.239   # fft analyza
dt = 0.01
fs = 1 / dt


for f in [freq,0.1,0.2,0.3,0.4,0.5,0.6,0.26, 0.22]:
    freqs = np.array([f])/fs
    scale = pywt.frequency2scale(wavelet, [freqs])
    coef, freqs = pywt.cwt(data, scale, wavelet,
                            sampling_period=dt)
    plt.plot(time, np.abs(coef)[0,:], label=f)
plt.legend()
plt.title(f"Waveletova transformace signalu pomoci vlnek\nzakladni frekvence je {freq}")

#%% Empirical Mode Demposition, Hilbert-Huang transform, https://emd.readthedocs.io
# https://www.mathworks.com/help/signal/ref/hht.html

import emd
sample_rate = dt

imf = emd.sift.sift(data)
IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert')
# Define frequency range (low_freq, high_freq, nsteps, spacing)
freq_range = (0.1, 10, 80, 'log')
f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False)

#%%

emd.plotting.plot_imfs(imf, time_vect=time, sharey=True)
#%%

emd.plotting.plot_imfs(imf, time_vect=time, sharey=False)

#%%
# fig = plt.figure(figsize=(10, 6))
# emd.plotting.plot_hilberthuang(hht, time, f,
#                                time_lims=(42, 50), freq_lims=(0.1, 15),
#                                fig=fig, log_y=True)

#%%
# config = emd.sift.get_config('sift')
# # Extract envelope options
# env_opts = config['envelope_opts']

# # Compute upper and lower envelopes
# upper_env = emd.sift.interp_envelope(data, mode='upper', **env_opts)
# lower_env = emd.sift.interp_envelope(data, mode='lower', **env_opts)
# # Compute average envelope
# avg_env = (upper_env+lower_env) / 2

# # Visualise
# plt.figure(figsize=(12, 6))
# plt.plot(data, 'k')
# plt.plot(upper_env, 'r')
# plt.plot(lower_env, 'b')
# plt.plot(avg_env, 'g')
# plt.legend(['Signal', 'Upper Envelope', 'Lower Envelope', 'Avg. Envelope'])
# #%%

# freq_range = (0.05, 20, 80)
# hht_f, spec = emd.spectra.hilberthuang(IF, IA, freq_range, scaling='density')
# plt.pcolormesh(time, hht_f, hht, cmap='hot_r')
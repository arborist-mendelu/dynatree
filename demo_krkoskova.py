#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:42:48 2024

@author: marik
"""
#%%

import krkoskova.lib_krkoskova as lk
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import signal
import numpy as np
from scipy.fft import fft, fftfreq
import pandas as pd


m = lk.Tuk("LP12", "tuk02")


#%%

out = [{'welch': i.welch(7), 'fft':i.fft, 'start':i.time[0], 'end':i.time[-1]} for i in m.signal_on_intervals("A03_y")]



fig, ax = plt.subplots()
for _ in out:
    i = _['welch']
    start = _['start']
    end = _['end']
    ax.plot(i.index, i.values, label=f"{start:.2f}-{end:.2f}")
    print(len(i.index))

ax.set(yscale='log', title="Welch spectrum")
ax.legend(title="Time interval")
ax.grid()
    
plt.show()
#%%

pd.DataFrame({i: out[i]['welch'].values.reshape(-1) for i in range(len(out))},
             index = out[0]['welch'].index)

#%%

fig, ax = plt.subplots()
for _ in out:
    i = _['fft']
    start = _['start']
    end = _['end']
    ax.plot(i.index, i.values, label=f"{start:.2f}-{end:.2f}")
    print(len(i.index))

ax.set(yscale='log', title="FFT spectrum")
ax.legend(title="Time interval")
ax.grid()
    
plt.show()


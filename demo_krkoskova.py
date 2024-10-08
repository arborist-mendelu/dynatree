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
import rich

m = lk.Tuk("LP15", "tuk01")
# rich.inspect(m)

#%%


sensor = "A01_z"
si = m.signal_on_intervals(sensor)
[i.restrict_signal(length=.05) for i in si]

for i in si:
    fig, ax = i.plot()
    plt.show()

#%%

df = [i.fft for i in si]
df = pd.concat(df, axis=1)
df.plot()
plt.show()
#%%

fig, ax = plt.subplots()

for i in si:
    print(i)
    i.fft.plot(ax=ax)
    ax.set(yscale='log', ylim=(0.01,None))
    plt.show()


#%%

s = lk.Signal(m.data_acc.loc[:,sensor], m.data_acc.index, m.dt(sensor), sensor)

# rich.inspect(m.peaks)

for start in m.peaks:
    delta = 0.1
    s.data.plot()
    plt.show()

#%%

plt.plot(s.data)
plt.show()

#%%
data = s.data.values
data = np.minimum(1,data)
data = np.maximum(-1,data)

f, t, Sxx = signal.spectrogram(data, s.fs, nperseg=2**11)
fig, ax = plt.subplots(figsize=(15,8))
ax.pcolormesh(t, f, Sxx, shading='gouraud')
ax.set( ylabel ='Frequency [Hz]', xlabel='Time [sec]')
# plt.ylim([0, 1000])
plt.show()

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


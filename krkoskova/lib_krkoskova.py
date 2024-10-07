#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:36:57 2024

@author: marik
"""

from scipy.signal import find_peaks
from scipy import signal
from scipy.fft import fft, fftfreq
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import solara

data_directory = "../data/krkoskova"
sensors = ["ext01", "ext02"] + ['A01_z', 'A01_y', 'A02_z', 'A02_y', 'A03_z', 'A03_y']

class Measurement:
    def __init__(self, tree, measurement):
        self.tree = tree
        self.measurement = measurement
        
    @property
    def data_extenso(self):
        """
        Returns dataframe with data from extensometers. Sampling is 500Hz.
        """
        try:
            df = pd.read_parquet(f"{data_directory}/{self.tree}_{self.measurement}_ext.parquet")
        except:
            return pd.DataFrame()
        df.index = np.arange(0,len(df))*0.002
        df.index.name = "Time"
        return df

    @property
    def data_acc(self):
        """
        Returns dataframe with data from accelerometers. Sampling is 5000Hz.
        """
        try:
            df = pd.read_parquet(f"{data_directory}/{self.tree}_{self.measurement}_acc.parquet")
        except:
            return pd.DataFrame()
        df.index = np.arange(0,len(df))*0.0002
        df.index.name = "Time"
        return df
    
    @property
    def release_time(self, probe="ext02"):
        if "tuk" in self.measurement:
            return 0
        df = self.data_extenso
        if len(df) == 0:
            return 0
        df = df[probe]
        df = df - df.mean()
        df = df.abs()
        release_time = df.idxmax()
        return release_time
    
    def sensor_sampling(self, sensor):
        if "ext" in sensor:
            return 500
        else:
            return 5000

    def dt(self, sensor):
        if "ext" in sensor:
            return 0.002
        else:
            return 0.0002

    def sensor_data(self, sensor):
        if "ext" in sensor:
            df = self.data_extenso
        else:
            df = self.data_acc
        if sensor in df.columns:
            return df[sensor]
        else:
            return pd.Series()
        
    def signal(self, sensor):
        data = self.sensor_data(sensor)[self.release_time:]
        return Signal(data.values, data.index, self.dt(sensor))
    
class Signal():
    def __init__(self, data_, time_, dt, extend=None):
        if extend is not None:
            time = np.arange(time_[0], time_[0]+extend,dt)
            data = np.interp(time, time_, data_, right=0)
        else:
            data,time = data_, time_
        self.data = data
        self.time = time
        self.dt = dt  
        if dt == 0.002:
            self.fs = 500
        elif dt == 0.0002:
            self.fs = 5000
            
    @property
    def tukey(self):
        sig = self.data
        sig = sig - sig.mean()
        tukey_window = signal.windows.tukey(len(sig), alpha=0.1, sym=False)
        return sig * tukey_window
        
    @property
    def fft(self):
        N = len(self.data)  # get the number of points
        xf_r = fftfreq(N, self.dt)[:N//2]
        data = self.data
        data = data - np.mean(data)
        try:
            data = data.values
        except:
            pass
        yf = fft(data)  # preform FFT analysis
        yf_r = 2.0/N * np.abs(yf[0:N//2])
        df_fft = pd.DataFrame(data=yf_r, index=xf_r)
        return df_fft

    def welch(self,n=8):
        nperseg = 2**n
        f, Pxx = signal.welch(x=self.data, fs=self.fs, nperseg=nperseg)
        df_welch = pd.DataFrame(index=f, data=Pxx)
        return df_welch
    
class Tuk(Measurement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        peaks,_ = (self.data_acc["A01_z"]
            .abs()
            .pipe(find_peaks, height=10, distance=1000)
        )
        self.peaks = self.data_acc["A01_z"].index[peaks]
    
    def signal_on_intervals(self,sensor):
        d = self.sensor_data(sensor)
        return [
            Signal(d[i:j], d[i:j].index, self.dt(sensor)) for i,j in zip(self.peaks[:-1], self.peaks[1:])
            ]

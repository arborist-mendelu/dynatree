from scipy.fft import fft, fftfreq
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import dynatree.dynatree


class SignalTuk():

    def __init__(self, parent, start, end, probe, accdata=None):
        self.parent = parent
        self.start = start
        self.end = end
        if accdata is None:
            dynatree.dynatree.logger.debug("Loading acc data from parent")
            self.accdata = parent.data_acc5000
        else:
            dynatree.dynatree.logger.debug("Loading acc data from parameter")
            self.accdata = accdata
        if (start > 0) & (end < self.accdata.index[-1]-0.45):
            self.signal = self.accdata.loc[start:end, probe].copy()
        else:
            zero_signal_start = pd.Series(index=np.arange(-1, 0, 0.0002), data=0)
            zero_signal_end = pd.Series(
                index=np.arange(self.accdata.index[-1]+0.002,
                self.accdata.index[-1]+1, 0.0002), data=0)
            elongated_signal =  pd.concat([zero_signal_start,self.accdata.loc[start:end, probe], zero_signal_end])
            self.signal = elongated_signal.loc[round(start-0.0001,4):round(end,4)].copy()
            self.signal.index = self.signal.index.round(4)
        self.dt = 0.0002
        self.probe = probe

    @property
    def fft(self):
        N = self.signal.shape[0]  # get the number of points
        xf_r = fftfreq(N, self.dt)[:N // 2]
        data = self.signal.values
        # data = data - data.mean()
        yf = fft(data)  # preform FFT analysis
        yf_r = 2.0 / N * np.abs(yf[0:N // 2])
        df_fft = pd.Series(index=xf_r, data=yf_r, name=self.probe)
        return df_fft


def find_peak_times_channelA(m, probe="a02_z", threshold=10):
    df = m.data_acc5000.loc[:, probe]
    if m.measurement != "M01":
        df = df.loc[:20]
    if probe=="a02_z":
        peaks, _ = find_peaks(df, threshold=threshold, distance=75)
    else:
        maximum = df.dropna().max()
        peaks, _ = find_peaks(df, prominence=maximum*0.75, distance=75)
    peak_times = df.index[peaks]
    peak_times = [round(i,2) for i in peak_times]
    return peak_times

def find_peak_times_channelB(m):
    return find_peak_times_channelA(m, probe="a02_x", threshold=6)

channelA = ["a01_x", "a01_y", "a01_z", "a02_y", "a02_z", "a03_x", "a03_y", "a03_z"]
channelB = ["a02_x", "a04_x", "a04_y", "a04_z"]

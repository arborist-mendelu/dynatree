from scipy.fft import fft, fftfreq
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class SignalTuk():

    def __init__(self, parent, start, end, probe):
        self.parent = parent
        self.start = start
        self.end = end
        self.signal = parent.data_acc5000.loc[start:end, probe]
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


def find_peak_times(m):
    df = m.data_acc5000.loc[:, "a02_z"]
    peaks, _ = find_peaks(df.abs(), threshold=10, distance=75)
    peak_times = df.index[peaks]
    return peak_times

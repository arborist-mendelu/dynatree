from dynatree.FFT import DynatreeSignal, df_failed_FFT_experiments
import numpy as np
from scipy.signal import hilbert
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import pywt
import pandas as pd
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go
from dynatree.dynatree import timeit
import time
from dynatree.dynatree import logger
import logging
from scipy.signal import decimate
import rich
import config
logger.setLevel(logging.ERROR)


class DynatreeDampedSignal(DynatreeSignal):

    def __init__(self, *args, damped_start_time=None, **kwargs):
        super().__init__(*args, **kwargs)
        data = self.signal_full
        data = data.dropna()
        data = data - data.iloc[0]
        if damped_start_time is not None:
            start_signal = damped_start_time
        else:
            start_signal = self.release_time
        data = data.loc[start_signal:]
        if data.iloc[0] < 0:
            data = data * (-1)
        # Najdi index první záporné hodnoty
        first_negative_index = data[data < 0].index[0]
        # Ořízni Series od první záporné hodnoty do konce
        data = data[first_negative_index:]
        data = data - data.mean()
        # TODO: Pro akcelerometry se bere samplovani taky po 0.01? Jinak je Hilberova obalka i wavelet i
        #       metoda pomoci maxim nepouzitelna.
        if "a0" in self.signal_source:
            self.dt = 0.01
            df = pd.DataFrame(decimate(data.values, 50))
            df.index = np.arange(0, len(df))*0.01 + data.index[0]
            data = df.copy()
        self.damped_data = data
        self.damped_signal = data.values.reshape(-1)
        self.damped_time = data.index

    @property
    def marked_failed(self):
        m = self.measurement
        coords = [m.measurement_type, m.day, m.tree, m.measurement,
                  self.signal_source]
        test_failed = coords in df_failed_FFT_experiments.values.tolist()
        logger.info(f"testing if failed {coords}")
        return test_failed


    @property
    @timeit
    def hilbert_envelope(self):
        amplitude_envelope = np.abs(hilbert(self.damped_signal))
        k, q = np.polyfit(self.damped_time[:-1],
                          np.log(amplitude_envelope[:-1]), 1)
        return {'data': amplitude_envelope, 'k': k, 'q': q}

    # @property
    @timeit
    def fit_maxima(self, maxpoints = np.inf):
        distance = 50
        window_length = 100
        if self.signal_source in ["Elasto(90)","blueMaj","yellowMaj"]:
            distance = 4
        if "a0" in self.signal_source:
            pass
            # distance = 50*100
            # window_length = 100*50
        smooth_signal = savgol_filter(self.damped_signal, window_length, 2)
        peaks, _ = find_peaks(np.abs(smooth_signal), distance=distance)
        peaks, _ = find_peaks(np.abs(self.damped_signal), distance=distance)
        # peaks = peaks.iloc[:maxpoints]

        k, q = np.polyfit(self.damped_time[peaks],
                          np.log(np.abs(self.damped_signal[peaks])), 1)

        return {'peaks': self.damped_data.iloc[peaks], 'k': k, 'q': q}

    @property
    @timeit
    def damped_signal_interpolated(self):
        """
        Returns interpolated damped signal with zero mean value
        and NOT multiplied by tukey window.
        """
        if self.signal_full is None:
            return
        signal = self.damped_signal
        time = self.damped_time

        # signal = signal.dropna()
        signal = signal - signal.mean()

        newindex = np.arange(time[0], time[-1] + self.dt, self.dt)
        newdata = np.interp(newindex, time, signal.reshape(-1), right=0)

        signal = pd.Series(index=newindex, data=newdata, name=self.signal.name)

        return signal

    @property
    @timeit
    def wavelet_envelope(self):
        start = time.time()
        wavelet = "cmor1-1.5"
        data = self.damped_signal_interpolated

        N = self.damped_signal_interpolated.shape[0]  # get the number of points
        xf_r = fftfreq(N, self.dt)[:N // 2]
        yf = fft(self.damped_signal_interpolated.values)  # preform FFT analysis
        yf_r = 2.0 / N * np.abs(yf[0:N // 2])
        df_fft = pd.Series(index=xf_r, data=yf_r, name=self.damped_signal_interpolated.name)
        freq = xf_r[yf_r.argmax()]
        # logger.info(f"FFT finished in {time.time() - start}, peak is at frequency {freq} Hz")

        def normalizace_waveletu(freq=0.2, dt=0.01):
            time = np.arange(0, 120, dt)
            data = np.cos(2 * np.pi * freq * time)

            scale = pywt.frequency2scale(wavelet, [freq * dt])
            coef, freqs = pywt.cwt(data, scale, wavelet,
                                   sampling_period=dt)
            coef = np.abs(coef)[0, :]
            return coef.max()

        dt = self.dt
        wavlet_norm = normalizace_waveletu(freq=freq, dt=dt)

        scale = pywt.frequency2scale(wavelet, [freq * dt])
        coef, freqs = pywt.cwt(data, scale, wavelet,
                               sampling_period=dt)
        coef = coef.reshape(-1)
        # logger.info(f"data.shape = {data.shape}, coef.shape = {coef.shape}")
        # logger.info(f"CWT finished in {time.time() - start}")
        # logger.info(f"wavelet normalize finished in {time.time() - start}, the norm is {wavlet_norm}")
        coef = np.abs(coef) / wavlet_norm
        maximum = np.argmax(coef)
        # logger.info(f"""
        #     Coef normalization finished in {time.time() - start},
        #     maximal coefficient at {maximum}, data are {data.index[maximum:-maximum]}
        #     and coef values are {coef[maximum:-maximum]}
        #     """)
        try:
            k, q = np.polyfit(data.index[maximum:-maximum], np.log(coef[maximum:-maximum]), 1)
        except:
            k, q = 0, 0
        return {'data': pd.Series(coef, index=data.index), 'k': k, 'q': q, "freq": freq, 'fft_data': df_fft}


def get_measurement_table():
    from dynatree import find_measurements
    df = find_measurements.get_all_measurements(method='all', type='all')
    df = df[df["measurement"] != "M01"]

    probes = ["Elasto(90)", "blueMaj", "yellowMaj",
              "Pt3", "Pt4"] + [f"a0{i}_{j}" for i in [1, 2, 3, 4] for j in ["y","z"]]

    # Rozšíření tabulky df pro každou hodnotu 'probe'
    df_expanded = df.loc[df.index.repeat(len(probes))].copy()
    df_expanded['probe'] = probes * len(df)

    df_failed = pd.read_csv(config.file['FFT_failed'])
    # Odstranění řádků, které se již nachází v df_failed
    df_result = df_expanded.merge(df_failed, how='left', on=['day', 'type', 'tree', 'measurement', 'probe'],
                                  indicator=True)
    df_result ["failed"] = df_result["_merge"] == "both"
    df = df_result.drop("_merge", axis=1).copy()

    df = df_result.copy()
    df = df[~((df["probe"].isin(["Pt3","Pt4"])) & (df["optics"]==False))]
    return (df)

def main():
    df = get_measurement_table()
    print(df.head())

if __name__ == "__main__":
    main()

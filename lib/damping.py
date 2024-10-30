from lib_FFT import DynatreeSignal
import numpy as np
from scipy.signal import hilbert
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import pywt
import pandas as pd
from scipy.fft import fft, fftfreq
import plotly.graph_objects as go


class DynatreeDampedSignal(DynatreeSignal):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        data = self.signal_full
        data = data.dropna()
        data = data - data[0]
        data = data[self.release_time:]
        if data.iloc[0] < 0:
            data = data * (-1)
        # Najdi index první záporné hodnoty
        first_negative_index = data[data < 0].index[0]
        # Ořízni Series od první záporné hodnoty do konce
        data = data[first_negative_index:]
        data = data - data.mean()
        # return data
        self.damped_data = data
        self.damped_signal = data.values
        self.damped_time = data.index

    @property
    def hilbert_envelope(self):
        amplitude_envelope = np.abs(hilbert(self.damped_signal))
        k, q = np.polyfit(self.damped_time[:-1],
                          np.log(amplitude_envelope[:-1]), 1)
        return {'data': amplitude_envelope, 'k': k, 'q': q}

    @property
    def fit_maxima(self):
        distance = 50
        if "Elasto" in self.signal_source:
            distance = 5
        smooth_signal = savgol_filter(self.damped_signal, 100, 2)
        peaks, _ = find_peaks(np.abs(smooth_signal), distance=distance)

        k, q = np.polyfit(self.damped_time[peaks],
                          np.log(np.abs(self.damped_signal[peaks])), 1)

        return {'peaks': self.damped_data.iloc[peaks], 'k': k, 'q': q}

    @property
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
        newdata = np.interp(newindex, time, signal, right=0)

        signal = pd.Series(index=newindex, data=newdata, name=self.signal.name)

        return signal

    @property
    def wavelet_envelope(self):
        wavelet = "cmor1-1.5"
        data = self.damped_signal_interpolated

        N = self.damped_signal_interpolated.shape[0]  # get the number of points
        xf_r = fftfreq(N, self.dt)[:N // 2]
        yf = fft(self.damped_signal_interpolated.values)  # preform FFT analysis
        yf_r = 2.0 / N * np.abs(yf[0:N // 2])
        df_fft = pd.Series(index=xf_r, data=yf_r, name=self.damped_signal_interpolated.name)
        freq = xf_r[yf_r.argmax()]

        def normalizace_waveletu(freq=0.2, dt=0.01):
            time = np.arange(0, 120, dt)
            data = np.cos(2 * np.pi * freq * time)

            scale = pywt.frequency2scale(wavelet, [freq * dt])
            coef, freqs = pywt.cwt(data, scale, wavelet,
                                   sampling_period=dt)
            coef = np.abs(coef)[0, :]
            return coef.max()

        dt = self.dt
        scale = pywt.frequency2scale(wavelet, [freq * dt])
        coef, freqs = pywt.cwt(data, scale, wavelet,
                               sampling_period=dt)

        # fig, ax = plt.subplots()
        # time = s.damped_signal_interpolated.index
        # plt.plot(time, s.damped_signal_interpolated)

        coef = np.abs(coef)[0, :] / normalizace_waveletu(freq=freq, dt=dt)
        maximum = np.argmax(coef)
        # ax.plot(time, coef, label=freq)
        # ax.plot(time[maximum], coef[maximum], "o")
        # ax.set(title="Waveletova transformace signalu pomoci vlnek")

        try:
            k, q = np.polyfit(data.index[maximum:-maximum], np.log(coef[maximum:-maximum]), 1)
        except:
            k, q = 0, 0
        # if len(time)<2*maximum:
        #     print("Inteval too short for wavelets")
        #     k, q = 0, 0
        # # ax.set(yscale='log')

        # y = np.exp(k*time+q)
        # plt.fill_between(time, y, -y, where=(y > 0), color='lightblue', alpha=0.5)
        # ax.grid()
        # k,q

        # ax = df_fft.plot()
        # ax.set(xlim=(0,1), yscale='log', title=f"FFT {m}")
        return {'data': coef, 'k': k, 'q': q, "freq": freq, 'fft_data': df_fft}


def draw_signal_with_envelope(s, envelope=None, k=0, q=0, ):
    signal, time = s.damped_signal, s.damped_time
    fig = go.Figure()
    x = time
    y = np.exp(k * time + q)
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                             y=np.concatenate([y, -y[::-1]]),
                             fill='toself',
                             fillcolor='lightblue',
                             line=dict(color='lightblue'),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name='signal', line=dict(color='blue')))
    if envelope is not None:
        fig.add_trace(
            go.Scatter(x=time, y=envelope, mode='lines', name='envelope', line=dict(color='red'), legendgroup='obalka'))
        fig.add_trace(go.Scatter(x=time, y=-envelope, mode='lines', showlegend=False, line=dict(color='red'),
                                 legendgroup='obalka'))
    fig.update_layout(xaxis_title="Čas", yaxis_title="Signál")
    return fig
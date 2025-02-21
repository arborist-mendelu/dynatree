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
import config
from scipy.stats import linregress
from parallelbar import progress_map
import statistics
from dynatree.dynatree import DynatreeMeasurement

logger.setLevel(logging.ERROR)
# logger.setLevel(logging.INFO)


class DynatreeDampedSignal(DynatreeSignal):
    """
    Usage example:

    >>> from dynatree.dynatree import DynatreeMeasurement
    >>> from dynatree.damping import DynatreeDampedSignal
    >>> m = DynatreeMeasurement(day="2022-08-16", tree="BK08", measurement="M02")
    >>> sig = DynatreeDampedSignal(m, signal_source="Elasto(90)")
    >>> sig.hilbert_envelope['k']
    >>> sig.damped_data.plot()
    """
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
        data = data - data.mean()  # Fix the case when the sensor is shifted
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
        self.vertical_finetuning = False

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
        signal = self.damped_signal_interpolated.values
        time = self.damped_signal_interpolated.index
        peaks = self.fit_maxima()['peaks']

        out = {}
        if self.vertical_finetuning:
            candidates = np.linspace(-5,5)
        else:
            candidates = [0]
        for yshift in candidates:
            amplitude_envelope = np.abs(hilbert(signal+yshift))
            mask = (time > peaks.index[0]) & (time < peaks.index[-1])
            x = time[mask]
            y = amplitude_envelope[mask]
            try:
                k, q, R2, p_value, std_err = linregress(x, np.log(y))
            except:
                k, q, R2, p_value, std_err = [None] * 5
            out[yshift] = [k, q, R2, p_value, std_err]
        df = pd.DataFrame.from_dict(out).T
        yshift = df[2].idxmin()
        k,q,R2,p_value,std_err = out[yshift]


        return {'data': [x,y], 'k': k, 'q': q, 'R2': R2, 'p': p_value, 'std_err': std_err, 'yshift': yshift}

    # @property
    @timeit
    def fit_maxima(self, threshold = config.damping_threshold):
        # distance = 50
        # window_length = 100
        # if self.signal_source in ["Elasto(90)","blueMaj","yellowMaj"]:
        #     distance = 4
        # if self.signal_source in ["Pt3","Pt4"]:
        #     distance = 80
        # if "a0" in self.signal_source:
        #     pass
        #     distance = 50*100
            # window_length = 100*50
        # smooth_signal = savgol_filter(self.damped_signal, window_length, 2)
        # peaks, _ = find_peaks(np.abs(smooth_signal), distance=distance)
        # peaks, _ = find_peaks(np.abs(self.damped_signal), distance=distance)

        T = 1/self.main_peak
        start = self.damped_signal_interpolated.index[0]
        analyzed = self.damped_signal_interpolated[start+T:]
        maximum = max(abs(analyzed))
        distance = int(T/self.dt/2*0.75)
        peaks, _ = find_peaks(np.abs(analyzed), distance=distance)
        peaks_number = np.argmax(np.abs(analyzed.iloc[peaks])-threshold*maximum < 0)-1
        peaks = peaks[:peaks_number]

        out = {}
        if self.vertical_finetuning:
            candidates = np.linspace(-5,5)
        else:
            candidates = [0]
        for yshift in candidates:
            try:
                k, q, R2, p_value, std_err = linregress(
                    analyzed.iloc[peaks].index,
                    np.log(np.abs(analyzed.iloc[peaks].values+yshift))
                )
            except Exception as e:
                logger.error(f"Error in fit_maxima {e}. {self.measurement}")
                k, q, R2, p_value, std_err = [None] * 5
            out[yshift] = [k, q, R2, p_value, std_err]
        df = pd.DataFrame.from_dict(out).T
        yshift = df[2].idxmin()
        k,q,R2,p_value,std_err = out[yshift]

        return {'peaks': analyzed.iloc[peaks], 'k': k, 'q': q, 'R2': R2, 'p': p_value, 'std_err': std_err, 'yshift':yshift}

    def ldd_from_definition(self):
        peaks = self.fit_maxima()['peaks']
        # logger.setLevel(logging.INFO)
        logger.info(f"peaks: {peaks}")
        positive = peaks[peaks > 0]
        ans = list(np.log(positive.iloc[:-1].values / positive.iloc[1:].values))
        negative = peaks[peaks < 0]
        ans = list(np.log(negative.iloc[:-1].values / negative.iloc[1:].values)) + ans
        ans = [i for i in ans if i>0]
        ldd = statistics.median(ans)
        T = 2 * np.nanmean(peaks.index.diff())
        b = ldd / T
        logger.info(f"ANS {ans} MEDIAN {ldd} T {T} b {b}")
        answer = {'k':b, 'LDD':ldd, 'T':T}
        logger.info(f"answer: {answer}")
        return answer

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

        freq = self.main_peak
        df_fft = self.fft
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

        peaks = self.fit_maxima()['peaks']
        mask = (data.index > peaks.index[0]) & (data.index < peaks.index[-1])
        # logger.info(f"""
        #     Coef normalization finished in {time.time() - start},
        #     maximal coefficient at {maximum}, data are {data.index[maximum:-maximum]}
        #     and coef values are {coef[maximum:-maximum]}
        #     """)
        # import rich
        # rich.print("Data",data)
        # rich.print("Coeff",coef)
        # rich.print("Msk",mask)
        independent = data.index[mask]
        dependent = coef[mask]
        try:
            k, q, R2, p_value, std_err = linregress(
                independent, np.log(dependent)
            )
        except:
            k, q, R2, p_value, std_err = [None] * 5
        return {'data': pd.Series(dependent, index=independent), 'k': k, 'q': q, "freq": freq, 'fft_data': df_fft, 'R2': R2,
                'p': p_value, 'std_err': std_err}


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

    df = df[~((df["probe"].isin(["Pt3","Pt4"])) & (df["optics"]==False))]
    df = df[df["failed"] == False]
    df = df.drop("failed", axis=1)
    return (df)

def process_row(index):
    day, method, tree, measurement, probe = index
    try:
        m = DynatreeMeasurement(day=day, tree=tree, measurement=measurement, measurement_type=method)
        s = DynatreeDampedSignal(m, signal_source=probe)
        fit_M = s.fit_maxima()
        fit_H = s.hilbert_envelope
        fit_W = s.wavelet_envelope
        out = [fit_W['freq'], fit_W['data'].index[0], fit_W['data'].index[-1]]
        out = out + [i[j] for i in [fit_M, fit_H, fit_W]  for j in ['k', 'R2', 'p', 'std_err'] ]
    except:
        print(f"Fail. {m}")
        out = [None]*15
    return out

def process_row_definition(index):
    """
    Evaluates LDD from the definition. The period is taken from peak distance rather from the
    FFT analysis
    """
    day, method, tree, measurement, probe = index
    try:
        m = DynatreeMeasurement(day=day, tree=tree, measurement=measurement, measurement_type=method)
        s = DynatreeDampedSignal(m, signal_source=probe)
        fit_ldd = s.ldd_from_definition()
        # logger.setLevel(logging.INFO)
        out = [fit_ldd['k'], fit_ldd['LDD'], fit_ldd['T']]
        logger.info(f"OUTPUT :{out}")
    except:
        print("Fail.")
        out = [None]*3
    return out

def main():
    """
    Evaluate damping using several methods

    Three methods are based on linear regression. We fit a line to points which form envelope
    (obtained by hilbert or wavelet transformation) in log space. With maxima method we use the
    same approach for points obtained as maxima and minima.
    """
    df = get_measurement_table()
    df = df[df["probe"] == "Elasto(90)"].reset_index(drop=True)
    df = df.set_index(["day", "type", "tree", "measurement", "probe"])
    metody = ['maxima', 'hilbert', 'wavelet']

    columns = ["freq", "start", "end"] + [i + j for i in metody for j in ['_b', '_R2', '_p', '_std_err']]

    results = progress_map(process_row, df.index.values)
    results_def = progress_map(process_row_definition, df.index.values)

    output = pd.DataFrame(results, index=df.index, columns=columns)
    output_def = pd.DataFrame(results_def, index=df.index, columns=['b','LDD','T'])

    for i in metody:
        output[f"{i}_b"] = np.abs(output[f"{i}_b"])
        output[f"{i}_LDD"] = output[f"{i}_b"]/output['freq']

    return output, output_def


if __name__ == "__main__":
    df, df_def = main()
    df.to_csv(config.file['outputs/damping_factor'])
    df_def.to_csv(config.file['outputs/damping_factor_def'])

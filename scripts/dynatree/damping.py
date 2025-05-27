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

def save_failed(text, ini = False):
    filename = "../temp/failed_damping.html"
    try:
        if ini:
            with open(filename, "w") as f:
                f.write(text)
        else:
            with open(filename, "a") as f:
                f.write(text)
    except:
        pass

save_failed("<h1>Failed Damping</h1>", ini=True)
manual_signal_ends = pd.read_csv(config.file['damping_manual_ends'], skipinitialspace=True)
manual_signal_ends = manual_signal_ends.set_index(["measurement_type", "day", "tree", "measurement"])

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
    def __init__(self, *args, damped_start_time=None, damped_end_time=None, **kwargs):
        super().__init__(*args, **kwargs)
        data = self.signal_full
        data = data.dropna()
        data = data - data.iloc[0]
        if damped_start_time is not None:
            start_signal = damped_start_time
        else:
            start_signal = self.release_time
        data = data.loc[start_signal:]

        if damped_end_time is None:
            coords = (self.measurement.measurement_type, self.measurement.day, self.measurement.tree, self.measurement.measurement)
            if coords in manual_signal_ends.index:
                damped_end_time = manual_signal_ends.at[coords, "end_time"]

        if damped_end_time is not None:
            data = data.loc[:damped_end_time]
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
        start_peak_index = 1
        T = 1/self.main_peak

        out = {}
        if self.vertical_finetuning:
            candidates = np.linspace(-5,5)
        else:
            candidates = [0]
        for yshift in candidates:
            amplitude_envelope = np.abs(hilbert(signal+yshift))
            mask = (time > peaks.index[start_peak_index]) & (time < peaks.index[-1])
            x = time[mask]
            y = amplitude_envelope[mask]
            try:
                b, q, R, p_value, std_err = linregress(x, np.log(y))
            except:
                b, q, R, p_value, std_err = [None] * 5
            out[yshift] = [b, q, R, p_value, std_err]
        df = pd.DataFrame.from_dict(out).T
        yshift = df[2].idxmin()
        try:
            b,q,R,p_value,std_err = out[yshift]
        except:
            b, q, R, p_value, std_err = [None]*5
        b = np.abs(b) if b is not None else None
        LDD = b*T if b is not None else None

        return {'data': [x,y], 'b': b, 'q': q, 'R': R, 'p': p_value, 'std_err': std_err,
                'LDD':LDD, 'yshift': yshift}

    # @property
    @timeit
    def fit_maxima(self, threshold = config.damping_threshold):
        """
        Returns
        -------
        dict with keys 'peaks', 'b', 'q', 'R', 'p', 'std_err', 'LDD', 'yshift'
        """
        T = 1/self.main_peak
        start = self.damped_signal_interpolated.index[0]
        analyzed = self.damped_signal_interpolated[start+T/2:] # skip first peak after release
        maximum = max(abs(analyzed))
        distance = int(T/self.dt/2*0.75)
        peaks, _ = find_peaks(np.abs(analyzed), distance=distance)
        peaks_number = np.argmax(np.abs(analyzed.iloc[peaks])-threshold*maximum < 0)-1
        peaks = peaks[:peaks_number]
        if peaks_number <5:
            logger.warning(f"Fit maxima {self}. Only {peaks_number} peaks used.")

        out = {}
        if self.vertical_finetuning:
            candidates = np.linspace(-5,5)
        else:
            candidates = [0]
        for yshift in candidates:
            try:
                b, q, R, p_value, std_err = linregress(
                    analyzed.iloc[peaks].index,
                    np.log(np.abs(analyzed.iloc[peaks].values+yshift))
                )
            except Exception as e:
                logger.error(f"Error in fit_maxima {e}. {self.measurement}")
                b, q, R, p_value, std_err = [None] * 5
            out[yshift] = [b, q, R, p_value, std_err]
        df = pd.DataFrame.from_dict(out).T
        yshift = df[2].idxmin()
        b,q,R,p_value,std_err = out[yshift]
        b = np.abs(b) if b is not None else None
        LDD = b*T if b is not None else None
        return {
            'peaks': analyzed.iloc[peaks], 'b': b, 'q': q, 'R': R, 'p': p_value, 'std_err': std_err, 'LDD': LDD,
            'yshift':yshift}

    def ldd_from_definition(self, peaks_limit = None):
        peaks = self.fit_maxima()['peaks']
        if peaks_limit is not None and len(peaks) > peaks_limit:
            peaks = peaks.iloc[:peaks_limit]
        logger.debug(f"ldd from definition, peaks: {peaks}")
        positive = peaks[peaks > 0]
        ans = list(np.log(positive.iloc[:-1].values / positive.iloc[1:].values))
        negative = peaks[peaks < 0]
        ans = list(np.log(negative.iloc[:-1].values / negative.iloc[1:].values)) + ans
        ans = [i for i in ans if i>0]
        logger.debug(f"ldd from definition, ans: {ans}")
        if len(ans)==0:
            return {'b': np.nan, 'LDD': np.nan, 'T': np.nan, 'std_err': np.nan}
        ldd = statistics.median(ans)
        T = 2 * np.nanmean(peaks.index.diff())
        b = ldd / T
        # logger.info(f"ANS {ans} MEDIAN {ldd} T {T} b {b}")
        answer = {'b':b, 'LDD':ldd, 'T':T, 'std_err': np.std(ans), 'peaks': peaks}
        # logger.info(f"answer: {answer}")
        return answer

    def ldd_from_two_amplitudes(self):
        """
        Method with working name defmulti.
        """
        # try:
        peaks = self.fit_maxima()['peaks']
        # logger.setLevel(logging.INFO)
        logger.info(f"peaks: {peaks}")
        # # Version 1: use four consecutive amplitudes
        # # LDD = ln (  ( |y0| \pm |y1| )  / ( |y2| \pm |y3| )  )
        # try:
        #     differences = (peaks.values[::2] - peaks.values[1::2])
        # except:
        #     differences = (peaks.values[:-1:2] - peaks.values[1::2])
        # logger.info(f"differences: {differences}")
        # quotients = np.log(differences[:-1]/differences[1:])
        # # Version 2: use three consecutive amplitudes
        # # LDD = 2 * ln (  ( |y0| \pm |y1| )  / ( |y1| \pm |y2| )  )
        # # LDD = 2 * ln (  ( y0 - y1 )  / ( - y1  + y2 )  )
        a = peaks.values
        ans = [2 * np.log( np.abs(a[i] - a[i+1]) / np.abs(-a[i+1]+a[i+2]) ) for i in range(len(a)-2)]
        ans = [i for i in ans if i>0]
        ldd = statistics.median(ans)
        T = 2 * np.nanmean(peaks.index.diff())
        b = ldd / T
        logger.info(f"ANS {ans} MEDIAN {ldd} T {T} b {b}")
        y = np.log(np.abs(peaks.values))
        x = peaks.index
        yp = np.mean(y)
        xp = np.mean(x)
        q = yp + b * xp
        numerator = sum([(xi-xp)*(yi-yp) for xi,yi in zip(x,y)])
        denominator = np.sqrt( (sum([(xi-xp)**2 for xi in x])) * (sum([(yi-yp)**2 for yi in y])) )
        R = numerator / denominator
        answer = {'b':b, 'LDD':ldd, 'T':T, 'std_err': np.std(ans), 'peaks': peaks, 'q': q, 'R': R, 'n': len(ans)}
        logger.info(f"answer: {answer}")
        # except:
        #     answer = {'b':None, 'LDD':None, 'T':None}
        #     logger.error(f"Failed {self}")
        return answer

    def ldd_from_distances(self):
        """
        Evaluate LDD from distances between maxima and minima. The first two minima and the first two maxima are used.
        LDD = ln (  ( |y0| pm |y1| )  / ( |y2| pm |y3| )  )
        The first four peaks are used (two maxima and two minima).
        """
        # try:
        peaks = self.fit_maxima()['peaks']
        if len(peaks) < 4:
            answer = {'b': None, 'LDD': None, 'T': None, 'std_err': None}
            return answer
        peaks = peaks.iloc[:4]
        a = peaks.values
        ldd = np.log( np.abs(a[0] - a[1]) / np.abs(a[2]-a[3]) )
        T = 2 * np.nanmean(peaks.index.diff())
        b = ldd / T
        logger.info(f"MEDIAN {ldd} T {T} b {b}")
        answer = {'b':b, 'LDD':ldd, 'T':T, 'std_err': 0}
        logger.info(f"answer: {answer}")
        # except:
        #     answer = {'b':None, 'LDD':None, 'T':None}
        #     logger.error(f"Failed {self}")
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
        T = 1/freq
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
        start_peak_index = 1
        mask = (data.index > peaks.index[start_peak_index]) & (data.index < peaks.index[-1])
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
            b, q, R, p_value, std_err = linregress(
                independent, np.log(dependent)
            )
        except:
            b, q, R, p_value, std_err = [None] * 5
        b = np.abs(b) if b is not None else None
        LDD = b*T if b is not None else None

        return {'data': pd.Series(dependent, index=independent),
                'b': b, 'q': q, "freq": freq, 'fft_data': df_fft, 'R': R,
                'p': p_value, 'std_err': std_err, 'LDD': LDD}


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
        out = out + [i[j] for i in [fit_M, fit_H, fit_W]  for j in ['b', 'R', 'p', 'std_err', 'LDD'] ]
    except Exception as e:
        print(f"FAIL. {m} {e}")
        print(f"SEE THIS URL https://euler.mendelu.cz/gallery/static/images/utlum/{m.day}_{m.measurement_type}_{m.tree}_{m.measurement}.png")
        save_failed(f"""<h2>{m}</h2> 
        <img src="https://euler.mendelu.cz/gallery/static/images/utlum/{m.day}_{m.measurement_type}_{m.tree}_{m.measurement}.png">
        <img src="https://euler.mendelu.cz/api/draw_graph/?method={m.day}_{m.measurement_type}&tree={m.tree}&measurement={m.measurement}&probe=Elasto%2890%29&start=0&end=1000000000&format=png">
        """)
        out = [None]*18
    return out

def process_row_definition(index):
    """
    Evaluates LDD from the definition. The period is taken from peak distance rather from the
    FFT analysis
    """
    try:
        day, method, tree, measurement, probe = index
        m = DynatreeMeasurement(day=day, tree=tree, measurement=measurement, measurement_type=method)
        s = DynatreeDampedSignal(m, signal_source=probe)
    except Exception as e:
        logger.error(e)
        logger.error(f"Failed LDD: {m}")
        save_failed(f"""<h2>{m}, failed LDD</h2> 
        <img src="https://euler.mendelu.cz/gallery/static/images/utlum/{m.day}_{m.measurement_type}_{m.tree}_{m.measurement}.png">
        <img src="https://euler.mendelu.cz/api/draw_graph/?method={m.day}_{m.measurement_type}&tree={m.tree}&measurement={m.measurement}&probe=Elasto%2890%29&start=0&end=1000000000&format=png">
        """)
        out = [None] * 6
        return out

    try:
        ldd = s.ldd_from_definition()
    except Exception as e:
        logger.error(e)
        ldd = {'b':None, 'LDD':None, 'T':None, 'std_err':None}
        logger.error(f"Failed LDD from definition: {m}")
        save_failed(f"""<h2>{m}, failed LDD from definition</h2> 
        <img src="https://euler.mendelu.cz/gallery/static/images/utlum/{m.day}_{m.measurement_type}_{m.tree}_{m.measurement}.png">
        <img src="https://euler.mendelu.cz/api/draw_graph/?method={m.day}_{m.measurement_type}&tree={m.tree}&measurement={m.measurement}&probe=Elasto%2890%29&start=0&end=1000000000&format=png">
        """)

    try:
        ldd2 = s.ldd_from_definition(peaks_limit=3)
    except Exception as e:
        logger.error(e)
        ldd2 = {'b':None, 'LDD':None, 'T':None, 'std_err':None}
        logger.error(f"Failed LDD from definition for just two peaks: {m}")
        save_failed(f"""<h2>{m}, failed LDD from definition for just two peaks</h2> 
        <img src="https://euler.mendelu.cz/gallery/static/images/utlum/{m.day}_{m.measurement_type}_{m.tree}_{m.measurement}.png">
        <img src="https://euler.mendelu.cz/api/draw_graph/?method={m.day}_{m.measurement_type}&tree={m.tree}&measurement={m.measurement}&probe=Elasto%2890%29&start=0&end=1000000000&format=png">
        """)

    try:
        ldd2diff = s.ldd_from_distances()
    except Exception as e:
        logger.error(e)
        ldd2diff = {'b':None, 'LDD':None, 'T':None, 'std_err':None}
        logger.error(f"Failed LDD from definition (from distances) for just two peaks: {m}")
        save_failed(f"""<h2>{m}, failed LDD from definition (from distances) for just two peaks</h2> 
        <img src="https://euler.mendelu.cz/gallery/static/images/utlum/{m.day}_{m.measurement_type}_{m.tree}_{m.measurement}.png">
        <img src="https://euler.mendelu.cz/api/draw_graph/?method={m.day}_{m.measurement_type}&tree={m.tree}&measurement={m.measurement}&probe=Elasto%2890%29&start=0&end=1000000000&format=png">
        """)

    try:
        ldd_multi = s.ldd_from_two_amplitudes()
    except Exception as e:
        logger.error(e)
        ldd_multi = {'b': None, 'LDD': None, 'T': None, 'std_err':None, 'R':None, 'n':None}
        logger.error(f"Failed LDD from multiple amplitudes: {m}")
        save_failed(f"""<h2>{m}, failed LDD from multiple amplitudes</h2> 
        <img src="https://euler.mendelu.cz/gallery/static/images/utlum/{m.day}_{m.measurement_type}_{m.tree}_{m.measurement}.png">
        <img src="https://euler.mendelu.cz/api/draw_graph/?method={m.day}_{m.measurement_type}&tree={m.tree}&measurement={m.measurement}&probe=Elasto%2890%29&start=0&end=1000000000&format=png">
        """)


    out = [
        ldd['b'], ldd['LDD'], ldd['T'],
        ldd2['b'], ldd2['LDD'], ldd2['T'],
        ldd2diff['b'], ldd2diff['LDD'], ldd2diff['T'],
        ldd_multi['b'], ldd_multi['LDD'], ldd_multi['T'], ldd_multi['R'], ldd_multi['n']
    ]
    logger.info(f"Dataset: {m}, {probe}, OUTPUT :{out}")
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

    columns = ["freq", "start", "end"] + [i + j for i in metody for j in ['_b', '_R', '_p', '_std_err', '_LDD']]

    results = progress_map(process_row, df.index.values)
    results_def = progress_map(process_row_definition, df.index.values)

    output = pd.DataFrame(results, index=df.index, columns=columns)
    output_def = pd.DataFrame(results_def, index=df.index, columns=['def_b','def_LDD','def_T',
                                                                    'def2_b', 'def2_LDD','def2_T',
                                                                    'def2diff_b', 'def2diff_LDD','def2diff_T',
                                                                    'defmulti_b','defmulti_LDD','defmulti_T','defmulti_R','defmulti_n'
                                                                    ])

    return output, output_def


if __name__ == "__main__":
    df, df_def = main()
    df = df.merge(df_def, how='left', left_index=True, right_index=True)
    df.to_csv(config.file['outputs/damping_factor'])


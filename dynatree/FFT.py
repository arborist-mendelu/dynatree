#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:02:32 2024

@author: marik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from gitdb.fun import chunk_size
from scipy.fft import fft, fftfreq
# import find_measurements
import matplotlib
import dynatree.multi_handlers_logger as mhl
import logging
import config
from dynatree import dynasignal, dynatree as dt
from dynatree import find_measurements
import gc
import time

from parallelbar import progress_map

# import resource
#
# # Nastavení limitu paměti (např. 500 MB)
# memory_limit = 5 * 1024 * 1024 * 1024  # v bajtech
# resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

length = 60  # the length of the signal
# todo: make min and max different for each tree
peak_min = .1 # do not look for the peak smaller than this value
peak_max = 0.7 # do not look for the peak larger than this value
df_manual_release_times = pd.read_csv(config.file["FFT_release"], index_col=[0,1,2,3,4])

class DynatreeSignal:

    def __init__(self, measurement, signal_source, release_source=None, dt=None, tukey=0.1):
        self.measurement = measurement
        self.signal_source = signal_source
        self.release_source = release_source
        if self.release_source is None:
            self.release_source = signal_source
        if self.measurement.data_pulling is not None and self.signal_source in self.measurement.data_pulling.columns:
            self.signal_full = self.measurement.data_pulling[self.signal_source]
            self.release_full = self.measurement.data_pulling[self.release_source]
        elif self.measurement.data_acc5000 is not None and self.signal_source in self.measurement.data_acc5000.columns:
            self.signal_full = self.measurement.data_acc5000[self.signal_source]
            self.release_full = self.measurement.data_acc5000[self.release_source]
        elif (self.signal_source in ["Pt3", "Pt4"]) & (self.measurement.data_optics_pt34 is not None):
            self.signal_full = self.measurement.data_optics_pt34[(self.signal_source, "Y0")]
            self.release_full = self.measurement.data_optics_pt34[(self.release_source, "Y0")]
        else:
            self.signal_full = None
            self.release_full = None
        if dt is not None:
            self.dt = dt
        elif "a0" in self.signal_source:
            self.dt = 0.0002
        else:
            self.dt = 0.01
        self.tukey = tukey
        self.manual_release_time = None

    @property
    def release_time(self):
        if self.manual_release_time is not None:
            return self.manual_release_time
        coords = (self.measurement.measurement_type, self.measurement.day, self.measurement.tree,
                  self.measurement.measurement, self.release_source)
        if coords in df_manual_release_times.index:
            return df_manual_release_times.at[coords,'release']
        data = self.release_full.dropna()
        data = data - data.iloc[0]
        data = data.dropna()
        release = data.loc[25:].abs().idxmax()
        return release
    
    @property
    def signal(self):
        """
        Returns interpolated signal with zero mean value
        multiplied by tukey window.
        """
        if self.signal_full is None:
            return
        signal = self.signal_full.loc[self.release_time:self.release_time+length]
        signal = signal.dropna()
        signal = signal - signal.mean()
        
        newindex = np.arange(signal.index[0], signal.index[0]+60+self.dt,self.dt)
        newdata = np.interp(newindex, signal.index, signal.values, right=0)
        
        signal = pd.Series(index=newindex, data=newdata, name=signal.name)
        tukey_window = scipy.signal.windows.tukey(len(signal), alpha=self.tukey, sym=False)
        signal = signal * tukey_window
        
        return signal
    
    @property 
    def fft(self):
        if self.signal_full is None:
            return
        N = self.signal.shape[0]  # get the number of points
        xf_r = fftfreq(N, self.dt)[:N//2]
        yf = fft(self.signal.values)  # preform FFT analysis
        yf_r = 2.0/N * np.abs(yf[0:N//2])
        df_fft = pd.Series(index=xf_r, data=yf_r, name=self.signal.name)
        return df_fft
    
    @property
    def main_peak(self):
        if self.signal_full is None:
            return
        return self.fft.loc[peak_min:peak_max].idxmax()
    
    def welch(self, nperseg=2**8):
        if self.signal_full is None:
            return
        if self.dt == 0.01:
            fs = 100
        if self.dt == 0.0002:
            fs = 5000
        if self.dt == 0.12:
            fs = 1/0.12
        return dynasignal.do_welch(pd.DataFrame(self.signal), nperseg=nperseg, fs=fs)
        

df_failed_FFT_experiments=pd.read_csv(config.file["FFT_failed"])
    
def process_one_probe(
        day='2021-03-22', tree='BK01', measurement='M03', measurement_type='normal', probe='Elasto(90)',
        # plot = 'never',
        plot = 'failed',
        ):
    """
    Parameter plot selects experiments for plot. Is suppoed to have values
    'never', 'failed' or 'all'
    """
    m = dt.DynatreeMeasurement(day=day, tree=tree, measurement=measurement, measurement_type=measurement_type)
    probename = probe
    release_source = probe
    if probe in ["blueMaj","yellowMaj"]:
        if m.file_pulling_name is None:
            return None
        probe = m.identify_major_minor[probe]
        release_source="Elasto(90)"
    test_failed = [f"{measurement_type}", f"{day}", f"{tree}", f"{measurement}", f"{probename}"] in df_failed_FFT_experiments.values.tolist()
    s = DynatreeSignal(m, probe, release_source=release_source)
    if test_failed:
        value = np.nan
    else:
        value = s.main_peak
    if plot=='never':
        del m
        return value 
    if plot=='failed' and not pd.isna(value):
        del m
        return value
    fig, ax = plt.subplots(2,1)
    sf = s.signal_full
    sf = sf - sf[0]
    sf.plot(ax=ax[0])
    s.signal.plot(ax=ax[0])
    if probe[0]=="a":
        ax[0].set(ylim=(s.signal.min(),s.signal.max()))
    s.fft.plot(logy=True, xlim=(0,3), ax=ax[1])
    ax[0].grid()
    ax[1].grid()
    ymax = s.fft.values.max()
    ax[1].set(ylim=(ymax/10**4,ymax*2), xlabel="Freq / Hz", ylabel="Amplitude")
    ax[0].set(xlim=(0,None), xlabel="Time / s", ylabel="Value")
    plt.suptitle(f"{s.measurement} {probename}, {s.main_peak:.03f} Hz".replace("Dynatree measurement ",""))
    if test_failed:
        prefix = "FAILED-"
    else:
        prefix = ""
        ax[1].axvline(s.main_peak, color='r', linestyle="--")
    plt.tight_layout()
    fig.savefig(f"../temp/fft_tukey/{prefix}{s.measurement.measurement_type}_{s.measurement.day}_{s.measurement.tree}_{s.measurement.measurement}_{probename}.png")
    plt.close('all')
    del m
    return value
    # print(s.measurement.measurement_type, s.measurement.day, 
    #       s.measurement.tree, s.measurement.measurement, s.signal_source,  s.main_peak)
    
    
# plot_one_probe(tree="BK04")    

# plot_one_probe(tree="BK09", measurement='M03', day="2021-03-22", probe="a03_z")    
    
#%%


def process_one_row(row):
    date, tree, measurement, measurement_type, optics, day, probe = row
    try:
        ansrow = process_one_probe(day, tree, measurement, measurement_type, probe=probe, plot='failed')
    except:
        msg = f"Spectral analysis failed for {date} {tree} {measurement} {measurement_type} {optics} {day} {probe}"
        print(msg)
        ansrow = None
    return [measurement_type, day, tree, measurement, probe, ansrow]

def process_df(df):
    n = 30  # chunk row size
    list_df = [df[i:i + n] for i in range(0, df.shape[0], n)]
    delka = len(list_df)
    i = 0
    start = time.time()
    ans = {}
    for _,d in enumerate(list_df):
        i = i+1
        print (f"*******   {i}/{delka}, runtime {time.time()-start} seconds ", end="\r")
        ans[_] = process_chunk(d)
    print(f"Finished in {round(time.time()-start)} seconds                           ")
    return ans

def process_chunk(df):
    ans = progress_map(process_one_row, df.values, disable=True)
    return ans


if __name__ == '__main__':
    # resource.setrlimit(resource.RLIMIT_DATA, (100 * 1024 * 1024, 100 * 1024 * 1024)) 
    
    logger = mhl.setup_logger(prefix="FFT_tukey_")
    logger.setLevel(logging.ERROR)
    logger.info("========== INITIALIZATION OF static-pull.py  ============")

    try:
        matplotlib.use('TkAgg')
    except:
        matplotlib.use('Agg')
    out = {}

    df = find_measurements.get_all_measurements(method='all', type='all')
    df = df[df["measurement"] != "M01"]

    probes = ["blueMaj", "yellowMaj", "Elasto(90)"]
    probes = probes + ["Pt3", "Pt4"]
    probes = probes + ["a01_z", "a02_z", "a03_z", "a04_z"]
    savecols = df.columns
    all = [list(j) + [i] for i in probes for j in df.values]
    df = pd.DataFrame(all, columns=list(savecols) + ["probe"])

    df = df[(~( (df["tree"] == "JD18") & (df["probe"].isin(["Pt3", "Pt4"])) ))]
    df = df[~( (df["probe"].isin(["Pt3", "Pt4"])) & (df["optics"] == False) )]
    df = df.reset_index(drop=True)
    # df = df.head(100)

    ans = process_df(df)

    ansdf = sum(ans.values(), start=[])
    ansdf = pd.DataFrame(ansdf, columns = ["type","day","tree","measurement","probe","peak"])
    ansdf.dropna().to_csv(f"../outputs/FFT_csv_tukey.csv", index=False)
    print(f"Saved FFT peaks, shape is {ansdf.shape}")

    # ans = process_one_row(["2021-06-29", "BK08", "M02", "normal", True, "2021-06-29", "a03_z"])
    # print(ans)
    # sys.exit()

    # input_data = [i for _, i in df.iterrows()]
    # ans = (progress_map(process_one_row, input_data))
    # df.loc[:,"peak"] = ans
    # df = df.loc[:,["type","day","tree","measurement","probe","peak"]]
    # df.to_csv(f"../outputs/FFT_csv_tukey.csv", index=False)

    # print(df.shape)
    # input_data = [i for _, i in df.iterrows()]
    # res = {}
    # for i in [1000,1500,2000,2500,3000,3500,4000,4025]:
    #     res[i] = progress_map(process_one_row, input_data[i:i+499])
    #     gc.collect()

    # # Paralelní zpracování
    # ans = []
    # with ProcessPoolExecutor(max_workers = 5) as executor:
    #     # Vytvoření úloh
    #     futures = {executor.submit(process_one_row, item): item for item in input_data}
    #
    #     # Sledování progresu pomocí tqdm
    #     for future in tqdm(as_completed(futures), total=len(futures), desc="Processing items"):
    #         ans.append(future.result())  # Uložení výsledků

    # with ProcessPoolExecutor() as executor:
    #     ans = list(tqdm(executor.map(process_one_row, input_data), total=len(input_data), desc="Processing items"))

    # print(input_data[1509])
    # process_one_row(input_data[1509])
    # for row in input_data:
    #     print(f"{row.name} ", end="", flush=True)
    #     process_one_row(row)

    # progress_map(process_one_row, input_data[:500], n_cpu=10)
    # df.loc[:,"peak"] = ans
    # df = df.loc[:,["type","day","tree","measurement","probe","peak"]]
    # df.to_csv(f"../outputs/FFT_csv_tukey.csv", index=False)


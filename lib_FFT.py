#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:02:32 2024

@author: marik
"""

import lib_dynatree as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy.fft import fft, fftfreq
import lib_find_measurements
from tqdm import tqdm
import resource
import matplotlib
import multi_handlers_logger as mhl
import logging


length = 60  # the length of the signal
peak_min = .1 # do not look for the peak smaller than this value
peak_max = 0.75 # do not look for the peak larger than this value

class DynatreeSignal:

    def __init__(self, measurement, signal_source, release_source=None, dt=0.0002, tukey=0.1):
        self.measurement = measurement
        self.signal_source = signal_source
        self.release_source = release_source
        if self.release_source is None:
            self.release_source = signal_source
        if self.signal_source in self.measurement.data_pulling.columns:
            self.signal_full = self.measurement.data_pulling[self.signal_source]
            self.release_full = self.measurement.data_pulling[self.release_source]
        elif self.signal_source in self.measurement.data_acc5000.columns:
            self.signal_full = self.measurement.data_acc5000[self.signal_source]
            self.release_full = self.measurement.data_acc5000[self.release_source]
        elif self.signal_source in ["Pt3", "Pt4"]:
            self.signal_full = self.measurement.data_optics_pt34[(self.signal_source, "Y0")]
            self.release_full = self.measurement.data_optics_pt34[(self.release_source, "Y0")]
        self.dt = dt
        self.tukey = tukey

    @property
    def release_time(self):
        data = self.release_full.dropna()
        data = data - data.iloc[0]
        data = data.dropna()
        release = data.loc[20:].abs().idxmax()
        return release
    
    @property
    def signal(self):
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
        N = self.signal.shape[0]  # get the number of points
        xf_r = fftfreq(N, self.dt)[:N//2]
        yf = fft(self.signal.values)  # preform FFT analysis
        yf_r = 2.0/N * np.abs(yf[0:N//2])
        df_fft = pd.Series(index=xf_r, data=yf_r, name=self.signal.name)
        return df_fft
    
    @property
    def main_peak(self):
        return self.fft.loc[peak_min:peak_max].idxmax()

df_failed_FFT_experiments=pd.read_csv("csv/FFT_failed.csv")
    
def plot_one_probe(day='2021-03-22', tree='BK01', measurement='M03', measurement_type='normal', probe='Elasto(90)'):
    m = dt.DynatreeMeasurement(day=day, tree=tree, measurement=measurement, measurement_type=measurement_type)
    probename = probe
    release_source = probe
    if probe in ["blueMaj","yellowMaj"]:
        probe = m.identify_major_minor[probe]
        release_source="Elasto(90)"
    s = DynatreeSignal(m, probe, release_source=release_source)
    fig, ax = plt.subplots(2,1)
    # print (s.release_time)
    sf = s.signal_full.copy()
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
    test = [f"{measurement_type}", f"{day}", f"{tree}", f"{measurement}", f"{probename}"] in df_failed_FFT_experiments.values.tolist()
    if test:
        prefix = "FAILED_"
        value = np.nan
    else:
        prefix = ""
        value = s.main_peak
        ax[1].axvline(s.main_peak, color='r', linestyle="--")
    plt.tight_layout()
    fig.savefig(f"../temp/fft_tukey/{prefix}{s.measurement.measurement_type}_{s.measurement.day}_{s.measurement.tree}_{s.measurement.measurement}_{probename}.png")
    plt.close('all')
    return value
    # print(s.measurement.measurement_type, s.measurement.day, 
    #       s.measurement.tree, s.measurement.measurement, s.signal_source,  s.main_peak)
    
    
# plot_one_probe(tree="BK04")    

# plot_one_probe(tree="BK09", measurement='M03', day="2021-03-22", probe="a03_z")    
    
#%%
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
    df = lib_find_measurements.get_all_measurements(method='all', type='all')  
    df = df[df["measurement"]!="M01"]
    
    
    probes = ["blueMaj", "yellowMaj", "Elasto(90)"]
    probes = probes + ["Pt3", "Pt4"]
    probes = probes + ["a01_z", "a02_z", "a03_z", "a04_z"]
    for probe in probes:
        print(f"Probe {probe}")
        pbar = tqdm(total=len(df))
        for i, row in df.iterrows():
            date, tree, measurement, measurement_type, optics, day = row
            pbar.set_description(f"{measurement_type} {day} {tree} {measurement}")
            if (tree=="JD18") & (probe in ["Pt3", "Pt4"]):
                pbar.update(1)
                continue
            try:
                out[(measurement_type, day, tree, measurement, probe)
                    ] = [plot_one_probe(day, tree, measurement, measurement_type, probe=probe)]
            except:
                logger.error(f"FFT failed for {measurement_type}, {day}, {tree}, {measurement}, {probe}")
            pbar.update(1)
        pbar.close()
    
    
    out_df = pd.DataFrame(out).T
    out_df = out_df.reset_index(drop=False)
    out_df.columns=["type","day","tree","measurement","probe","peak"]
    out_df.to_csv(f"../outputs/FFT_csv_tukey.csv", index=False)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 07:56:28 2023

@author: marik
"""

import pandas as pd
import numpy as np
from scipy import interpolate, signal
from scipy.fft import fft, fftfreq

def read_data(file, index_col="Time", usecols=None):
    """
    Reads csv file, returns dataframe.
    The index_col is the index of the column with index for the dataframe. 
    If index is string, then the position of the index is guessed from the first line
    of the csv file.
    """
    # If the index_col is string, find the position of index_col in the first line
    # and set index_col to this position. This allows to specify index_col for 
    # data with multiindex.
    print (f"Reading file {file}.")
    if isinstance(index_col,str):
        with open(file) as f:
            first_line = f.readline().strip('\n')
            index_col = first_line.split(",").index("Time")
    df = pd.read_csv(file,header=[0,1], index_col=index_col, dtype = np.float64)  # read the file
    if ("source","data") in df.columns: # drop unenecessary column
        df.drop([("source","data")],axis=1,inplace=True)  
    df["Time"] = df.index  # pro pohodlí, aby se k času dalo přistupovat i jako data.Time
    return df

def directory2date(d):
    """
    Converts directory from the form '01_Mereni_Babice_22032021_optika_zpracovani'
    to date like 2021-03-22
    """
    return f"{d[21:25]}-{d[19:21]}-{d[17:19]}"  

def filename2tree_and_measurement_numbers(f):
    tree,tree_measurement,*_ = f.split("_")
    tree = tree.replace("BK","")
    tree_measurement = tree_measurement.replace("M0","").replace(".csv","")
    return tree,tree_measurement


def find_release_time_optics(df,probe="Pt3",coordinate="Y0"):
    """
    Finds release time lokinng for maximal displacement from the 
    initial position. Probe and coordiante defined in the parameters are 
    used.
    Parameters
    ----------
    df : dataframe 
    probe : TYPE, optional
        DESCRIPTION. The default is "Pt3".
    coordinate : TYPE, optional
        DESCRIPTION. The default is "Y0".

    Returns
    -------
    Index of the release. If index is Time, returns time.
    """
    movement_data = df[(probe,coordinate)]
    movement_data = movement_data - movement_data[0]
    movement_data = np.abs(movement_data)
    return movement_data.idxmax(axis=0)

## Makra pro nalezeni Probu na BL shora dolu.
def probes(b, axis="Y", cam = 1):
    """
    Ouptuts the chain of probes on bendlines. BL 44, 52 and 60 are shorter (end
    points plus five points inside), the other are longer (10 points inside).
   

    Parameters
    ----------
    b : TYPE
        DESCRIPTION.
    axis : TYPE, optional
        DESCRIPTION. The default is "Y".
    cam : 1 pro kameru z boku (default), 0 pro kameru ve směru tahu    

    Returns
    -------
    out : List of 7 or 12 tuples with names of probes on bendline from top
    to the bottom.
    
    """
    coord = axis
    if cam==1:
        bls = [44,52,60]
    else:
        bls = [10,18,26]
    if b in bls:
        num = 5
    else:
        num = 10
    out = [(f"BL{b}",j) for j in [f"Pt0A{coord}", *[f"{coord}{i}" for i in range(num)], f"Pt0B{coord}"]]
    return out

def get_chains_of_bendlines(axis="Y", cam=1):
    if cam == 1:
        start = 44 # side view
    else:
        start = 10 # back view
    # Find probes on all bendlines
    A = [ probes(i,axis=axis,cam=cam) for i in range(start,start+24)]
    # Convert into one long list
    all = sum(A, start = [])
    # Split list into three parallel bendlines
    l = len(all)
    output = [all[i*l//3:(i+1)*l//3] for i in range(3)]
    return output

def do_fft(signal, time):
    time = time - time[0] # restart time from zero
    # signal = signal.values # grab values
    fs = 100
    time_fft = np.arange(time[0],time[-1],1/fs) # timeline for resampling
    f = interpolate.interp1d(time, signal, fill_value="extrapolate")  
    signal_fft = f(time_fft) # resample
    signal_fft = signal_fft - np.nanmean(signal_fft) # mean value to zero
    
    N = time_fft.shape[0]  # get the number of points
    yf = fft(signal_fft)  # preform FFT analysis
    xf_r = fftfreq(N, 1/fs)[:N//2]
    yf_r = 2.0/N * np.abs(yf[0:N//2])
    return xf_r,yf_r

def do_welch(s, time):
    time = time - time[0] # restart time from zero
    # signal = signal.values # grab values
    fs = 100
    time_welch = np.arange(time[0],time[-1],1/fs) # timeline for resampling
    f = interpolate.interp1d(time, s)  
    signal_welch = f(time_welch) # resample
    signal_welch = signal_welch - np.nanmean(signal_welch) # mean value to zero
    f, Pxx = signal.welch(x=signal_welch, fs=fs, nperseg=2**10)
    Pxx
    return f, Pxx

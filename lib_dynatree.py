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
    #print (f"Reading file {file}.")
    if isinstance(index_col,str):
        with open(file) as f:
            first_line = f.readline().strip('\n')
            index_col = first_line.split(",").index("Time")
    df = pd.read_csv(file,header=[0,1], index_col=index_col, dtype = np.float64)  # read the file
    if ("source","data") in df.columns: # drop unenecessary column
        df.drop([("source","data")],axis=1,inplace=True)  
    df["Time"] = df.index  # pro pohodlí, aby se k času dalo přistupovat i jako data.Time
    return df



def read_data_selected(file,
                        probes=["Time"] + [f"Pt{i}" for i in [0,1,3,4,8,9,10,11,12,13]]):
    """

    Parameters
    ----------
    file :  csv files with data
    probes : probes which should be included in te output
        DESCRIPTION. The default is ["Time"] + [f"Pt{i}" for i in [0,1,3,4,8,9,10,11,12,13]].

    Returns
    -------
    df :  dataframe from the file, but only columns with first index level specified in 
    probe variable. 
    """
    # find column numbers to read
    df_headers = pd.read_csv(file, 
                     nrows=2, header=None,
                     dtype=object
                     ).fillna('')
    first_row = df_headers.iloc[0,:].values
    seznam = np.nonzero(np.isin(first_row,probes))[0]
    sloupce = df_headers[seznam].values

    # read csv file    
    df = pd.read_csv(file, 
                     skiprows=1,
                     dtype=np.float64, 
                     usecols=seznam
                     )
    # adjust index and column names
    df.columns = pd.MultiIndex.from_arrays(sloupce)
    df.index = df["Time"].values.reshape(-1)
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

# Makra pro FFT a Welch
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

def do_welch(s, time, nperseg=2**10, fs = 100):
    time = time - time[0] # restart time from zero
    # signal = signal.values # grab values
    time_welch = np.arange(time[0],time[-1],1/fs) # timeline for resampling
    f = interpolate.interp1d(time, s)  
    signal_welch = f(time_welch) # resample
    signal_welch = signal_welch - np.nanmean(signal_welch) # mean value to zero
    f, Pxx = signal.welch(x=signal_welch, fs=fs, nperseg=nperseg)
    Pxx
    return f, Pxx

def read_data_inclinometers(file, release=None, delta_time=0):
    """
    Read data from pulling tests, restart Time from 0 and turn Time to index.
    If release is given, shift the Time and index columns so that the release 
    is at the given time. In this case the original time in in the column Time_inclino
    """
    df_pulling_tests = pd.read_csv(
        file,
        skiprows=55, 
        decimal=",",
        delim_whitespace=True,    
        skipinitialspace=True,
        na_values="-"
        )
    df_pulling_tests["Time"] = df_pulling_tests["Time"] - df_pulling_tests["Time"][0]
    df_pulling_tests.set_index("Time", inplace=True)
    # df_pulling_tests.interpolate(inplace=True, axis=1)
    if release is None:
        return df_pulling_tests
    
    if df_pulling_tests["Force(100)"].isna().all():
        release_time_force = release
    else:
        release_time_force = df_pulling_tests["Force(100)"].idxmax()
        
    # Sync the dataframe from inclino to optics    
    df_pulling_tests["Time_inclino"] = df_pulling_tests.index
    df_pulling_tests["Time"] = df_pulling_tests["Time_inclino"] - release_time_force + release + delta_time
    df_pulling_tests.set_index("Time", inplace=True)
    df_pulling_tests["Time"] = df_pulling_tests.index
        
    return df_pulling_tests

def find_finetune_synchro(date, tree, measurement, cols="delta time"):
    """
    Returns line from csv/synchronization_finetune_inclinometers_fix.csv
    corresponding to the date, tree and measurement numbers. Accepts both 
    24122021 and 2021-12-24 for date, both BK02 and 02 for tree, both 3 and M03 
    for measurement. 
    
    Raises error, is there are more than two rows in the file.
    Returns the value or the array corresponding to the col variable.
    
    If cols is "delta time" returns 0 if not found and the value is found.
    
    In ither cases returns None if the row is not found
    
    Date is either 22032021 or 2021-03-22 format or 01_Mereni_Babice_22032021_optika_zpracovani
    """
    if "Mereni_Babice" in date:
        date = directory2date(date)
    if not "-" in date:
        date = f"{date[-4:]}-{date[2:4]}-{date[:2]}"
    if not "BK" in str(tree):
        tree = f"BK{tree}"
    if not "M" in str(measurement):
        measurement = f"M0{measurement}"
    df = pd.read_csv("csv/synchronization_finetune_inclinometers_fix.csv",header=[0,1], index_col=[0,1,2])     
    if not (date,tree,measurement) in df.index:
        if cols=="delta time":
            return 0
        else:
            return None
    df = df.loc[(date,tree,measurement),cols]
    # if df.shape[0]>1:
    #     raise Exception (f"Row {date} {tree} {measurement} is more than once in the file csv/synchronization_finetune_inclinometers_fix.csv.\nMerge the date into a single row." )
    output = df.values
    if len(output)==1:
        output = output[0]

    if cols=="delta time":
        if output is None or np.isnan(output):
            output = 0
        
    return output

# find_finetune_synchro("2021-03-22", "BK01", "M02")
# find_finetune_synchro("2022-04-05", "BK01", 3)
start,end = find_finetune_synchro("2021-03-22", "BK01", "M02", "Inclino(80)X")
start,end = find_finetune_synchro("2021-03-22", "BK01", "M02", "Inclino(80)Y")


def date2dirname(date):
    # accepts all "22032021", "2021-03-22" and "01_Mereni_Babice_22032021_optika_zpracovani" as measurement_day
    if len(date)==10:
        date = "".join(reversed(date.split("-")))
    if len(date)==8:
        date = f"01_Mereni_Babice_{date}_optika_zpracovani"
    return date


def find_release_time_interval(df_extra, date, tree, measurement):
    """
    Find release time
    
    Returns manually determined time limits if there are data available in the csv file.
    Returns [0,0]
    
    Othervise returns time interval in which trhe force is between 80 and 95 percent 
    of maxima.
    """

    
    check_manual_data = find_finetune_synchro("2021-03-22", "BK01", "M02", cols="pre_release")
    if check_manual_data is not None and ~(np.isnan(check_manual_data).any()):
        return check_manual_data
    
    if df_extra["Force(100)"].isna().values.all():
        return [0,0] 
    else:
        maxforceidx = df_extra["Force(100)"].idxmax().iat[0]
        maxforce  = df_extra["Force(100)"].max().iat[0]
        percent1 = 0.95
        tmax = np.abs(df_extra.loc[:maxforceidx,["Force(100)"]]-maxforce*percent1).idxmin().values[0]
        percent2 = 0.85
        tmin = np.abs(df_extra.loc[:maxforceidx,["Force(100)"]]-maxforce*percent2).idxmin().values[0]
        return tmin,tmax





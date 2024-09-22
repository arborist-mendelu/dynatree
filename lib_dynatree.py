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
import glob
import os
from functools import wraps, lru_cache, cached_property
import time

# 
import logging
from logging.handlers import RotatingFileHandler
logFile = '/tmp/dynatree.log'

try:
    logger = logging.getLogger("lib_dynatree")
    file_handler = RotatingFileHandler(logFile, maxBytes=100000, backupCount=10)
    screen_handler = logging.StreamHandler()
    logging.basicConfig(
        handlers=[
            file_handler,
            screen_handler
        ],
        level=logging.ERROR,
        format="[%(asctime)s] %(levelname)s | %(message)s",
    )
    file_handler.setLevel(logging.ERROR)
except:
    logger = logging.getLogger("lib_dynatree")
    screen_handler = logging.StreamHandler()
    logging.basicConfig(
        handlers=[
            screen_handler
        ],
        level=logging.ERROR,
        format="[%(asctime)s] %(levelname)s | %(message)s",
    )
    logger.error("Log file not available for writing. Logs are on screen only.")

df_scale_factors = pd.read_csv(
    "csv/scale_factors.csv", index_col=[0,1,2]
    ).sort_index()
df_reset_inclinometers = pd.read_csv(
    "csv/reset_inclinometers.csv", index_col=[0,1,2,3]
    ).sort_index()


def get_notes():
    """
    Je potreba opracvit pripady, kdy misto Measurement je Measurement after 
    RO nebo neco podobneho.
    """
    notes = pd.read_csv("csv_output/measurement_notes.csv", index_col=0)
    notes['day'] = notes['day'].str.split("_").str[0]
    
    # # Přidání nuly ve sloupci 'measurement' (M1 -> M01)
    notes['measurement'] = notes['measurement'].str.replace('M(\d)', lambda x: f"M0{x.group(1)}", regex=True)
    
    # # Transformace sloupce 'tree' (1 -> BK01)
    notes['tree'] = notes['tree'].astype(str).str.zfill(2)
    notes['tree'] = 'BK' + notes['tree']
    
    notes = notes.set_index(['day', 'measurement', 'tree'])
    notes = notes.dropna(how='all').fillna("")
    notes["remark"] = notes["remark1"] + " " + notes["remark2"]
    return notes

# notes = get_notes()

def tand(angle):
    """
    Evaluates tangens of the angle. The angli is in degrees.
    """
    return np.tan(np.deg2rad(angle))

def arctand(value):
    """
    Evaluates arctan. Return the angle in degrees.
    """
    return np.rad2deg(np.arctan(value))    

# from https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        msg = f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f}s.'
        if len (msg)>400:
            msg = f'Function {func.__name__} Took {total_time:.4f}s.'
        logger.info(msg)
        return result
    return timeit_wrapper

# Property timer, https://stackoverflow.com/questions/64589914/how-to-decorate-a-property-to-measure-the-time-it-executes
def timer(method):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = method(*args, **kwargs)  # note the function call!
        end = time.time()
        logger.info(f'Elapsed time for {method.__name__} is {(end-start):.4f}s.')
        return result
    return wrapper

# @timeit
@lru_cache(3)
def read_data(file, index_col="Time", usecols=None):
    """
    If file is parquet file, return the dataframe from this file.
    
    The rest if relict from previous versions.
    Reads csv file, returns dataframe.
    The index_col is the index of the column with index for the dataframe. 
    If index is string, then the position of the index is guessed from the first line
    of the csv file.
    """
    # If the index_col is string, find the position of index_col in the first line
    # and set index_col to this position. This allows to specify index_col for 
    # data with multiindex.
    #print (f"Reading file {file}.")
    if "parquet" in file:
        return pd.read_parquet(file)
    if isinstance(index_col,str):
        with open(file) as f:
            first_line = f.readline().strip('\n')
            index_col = first_line.split(",").index("Time")
    df = pd.read_csv(file,header=[0,1], index_col=index_col, dtype = np.float64)  # read the file
    if ("source","data") in df.columns: # drop unenecessary column
        df.drop([("source","data")],axis=1,inplace=True)  
    df["Time"] = df.index  # pro pohodlí, aby se k času dalo přistupovat i jako data.Time
    return df
    
def get_data(date, tree, measurement):
    """
    Loads the data file corresponding to date, tree and measurement
    """
    file = f"../data/parquet/{date.replace('-','_')}/BK{tree}_M0{measurement}.parquet"
    return read_data(file)


def read_data_selected(file,
                        probes = ["Time", 'Pt0', 'Pt1', 'Pt3', 'Pt4', 'Pt8', 'Pt9', 'Pt10', 'Pt11', 'Pt12', 'Pt13']
                        ):
    # lru_cache vould complain TypeError: unhashable type: 'list'
    return read_data_selected_doit(file, tuple(probes))


# @timeit
@lru_cache(4)
def read_data_selected_doit(file, probes):
    """

    Parameters
    ----------
    file :  csv or parquet file with data
    probes : probes which should be included in te output
        DESCRIPTION. The default is ["Time"] + [f"Pt{i}" for i in [0,1,3,4,8,9,10,11,12,13]].

    Returns
    -------
    df :  dataframe from the file, but only columns with first index level specified in 
    probe variable. 
    """
    if "parquet" in file:
        df = pd.read_parquet(file)
        columns = [i for i in df.columns if i[0] in probes]
        df = df[columns]
        # remove possible nan strings in column names
        cols = [ (i[0],'' if i[1]=='nan' else i[1]) for i in df.columns]
        df.columns = pd.MultiIndex.from_tuples(cols)
        return df
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
    if "Time" in probes:
        df.index = df["Time"].values.reshape(-1)

    # remove possible nan strings in column names
    cols = [ (i[0],'' if i[1]=='nan' else i[1]) for i in df.columns]
    df.columns = cols
    return df

def directory2date(d):
    """
    Converts directory from the form '2021_03_22'
    to date like 2021-03-22
    """
    return d.replace("_","-")  


def filename2tree_and_measurement_numbers(f):
    tree,tree_measurement,*_ = f.split("_")
    tree = tree.replace("BK","")
    tree_measurement = tree_measurement.replace("M0","").replace(".parquet","")
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

def read_data_inclinometers(m, release=None, delta_time=0):
    """
    Read data from pulling tests, restart Time from 0 and turn Time to index.
    If release is given, shift the Time and index columns so that the release 
    is at the given time. In this case the original time is in the column Time_inclino
    
    m .. measurementm instance of DynatreeMeasurement
    
    """
    df_pulling_tests = m.data_pulling
    if "Time" not in df_pulling_tests.columns:
        df_pulling_tests["Time"] = df_pulling_tests.index
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
    # if delta_time != 0:
    #     print(f"  info: Using time fix {delta_time} when reading data from inclino/force/elasto")
    df_pulling_tests["Time_inclino"] = df_pulling_tests.index
    df_pulling_tests["Time"] = df_pulling_tests["Time_inclino"] - release_time_force + release + delta_time
    df_pulling_tests.set_index("Time", inplace=True)
    df_pulling_tests["Time"] = df_pulling_tests.index

    return df_pulling_tests

def find_finetune_synchro(date, tree, measurement, cols="delta time"):
    """
    Returns line from csv/synchronization_finetune_inclinometers_fix.csv
    corresponding to the date, tree and measurement numbers. Accepts both 
    2021_12_24 and 2021-12-24 for date, both BK02 and 02 for tree, both 3 and M03 
    for measurement. 
    
    Returns the value or the array corresponding to the col variable.
    
    If cols is "delta time" returns 0 if not found and the value if found.
    
    In all cases returns None if the row is not found
    
    Date is either 2021_03_22 or 2021-03-22 format 
    """
    date = date.replace("_","-")
    if not "BK" in str(tree):
        tree = f"BK{tree}"
    if not "M" in str(measurement):
        measurement = f"M0{measurement}"
    df = pd.read_csv("csv/synchronization_finetune_inclinometers_fix.csv",header=[0,1], index_col=[0,1,2])     
    df = df.sort_index()
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
# start,end = find_finetune_synchro("2021-03-22", "BK01", "M02", "Inclino(80)X")
# start,end = find_finetune_synchro("2021-03-22", "BK01", "M02", "Inclino(80)Y")


# def date2dirname(date):
#     # accepts all "22032021", "2021-03-22" and "01_Mereni_Babice_22032021_optika_zpracovani" as measurement_day
#     if len(date)==10:
#         date = "".join(reversed(date.split("-")))
#     if len(date)==8:
#         date = f"01_Mereni_Babice_{date}_optika_zpracovani"
#     return date


def find_release_time_interval(df_extra, date, tree, measurement):
    """
    Find release time
    
    Returns manually determined time limits if there are data available in the csv file.
    Returns [0,0] if the data for force do not exist.
    
    Othervise returns time interval in which the force is between 80 and 95 percent 
    of maxima.
    """

    check_manual_data = find_finetune_synchro(date, tree, measurement, cols="pre_release")
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

def split_path(file, suffix="parquet"):
    # fix for bad names containing BK_10 instead of BK10
    file = file.replace("BK_","BK") 
    data = file.split("/")
    data[-1] = data[-1].replace(f".{suffix}","")
    return [file,data[-2].replace("_","-")] + data[-1].split("_")

def get_all_measurements(cesta="../data", suffix="parquet", directory="parquet"):
    """
    Get dataframe with all measurements. The dataframe has columns
    date, tree and measurement.
    
    Intended to find measurements from optics
    """
    files = glob.glob(cesta+f"/{directory}/*/BK*M??.{suffix}") 
    files += glob.glob(cesta+f"/{directory}/*/BK*M?.{suffix}")
    out = [split_path(file, suffix=suffix) for file in files]
    df = pd.DataFrame([i[1:] for i in out], columns=['day','tree', 'measurement'])
    df = df.sort_values(by=list(df.columns))
    df = df.reset_index(drop=True)
    return df



date2color = {
    '2022-04-05': "C0",
    '2022-08-16': "C1",
    '2021-03-22': "C0",
    '2021-06-29': "C1",
    }

def fix_inclinometers_sign(df_, measurement_type, day, tree):
    """
    Fix the sign of the inclinometer. The function is used to ensure that 
    major axis has positive values of inclination.
    """
    df_scale_factors = pd.read_csv("csv/scale_factors.csv", index_col=[0,1,2]
                                   ).sort_index()
    df = df_.copy()
    coords = (measurement_type,day,tree)
    if coords in df_scale_factors.index:
        df_temp = df_scale_factors.loc[coords,:]
        for i,row in df_temp.iterrows():
            df[row['probe']+"X"] = df[row['probe']+"X"] * row['scale_factor']
            df[row['probe']+"Y"] = df[row['probe']+"Y"] * row['scale_factor']
    return df

class DynatreeMeasurement:
    def __init__(self, day, tree, measurement, measurement_type='normal', datapath="../data"):
        if not isinstance(tree, str):
            tree = f"{tree:02}"
        if not isinstance(measurement, str):
            measurement = f"{measurement}"
        tree = tree[-2:]
        if tree == "18":
            tree = "JD"+tree
        else:
            tree = "BK"+tree
        self.tree = tree
        self.day = day.replace("_","-")
        measurement = f"M0{measurement[-1]}"
        self.measurement = measurement
        self.measurement_type = measurement_type.lower()
        self.datapath = datapath
        self.date = self.day
            
    def __str__(self):
        return (f"[Dynatree measurement {self.day} {self.tree} {self.measurement} {self.measurement_type}]")
        
    def __repr__(self):
        return (f"[Dynatree measurement {self.day} {self.tree} {self.measurement} {self.measurement_type}]")
        
    @property
    def file_pulling_name(self):
        """
        The name of the file with Force+Elasto+Inclino.
        Raw data from the device converted from txt files to parquet.
        """
        return f"{self.datapath}/parquet_pulling/{self.day.replace('-','_')}/{self.measurement_type}_{self.tree}_{self.measurement}.parquet"
    @property
    def file_optics_name(self):
        """
        The name of the file with optics.
        """
        if self.measurement_type != "normal":
            logger.warning(f"Optics not available for {self.day} {self.tree} {self.measurement} {self.measurement_type}")
            return ""
        return f"{self.datapath}/parquet/{self.day.replace('-','_')}/{self.tree}_{self.measurement}.parquet"
    
    @property
    def file_optics_extra_name(self):
        """
        Returns the file with Force, Elasto, Inclino. The file is 
        interpolated to the same index as optics and both dataframes
        can be concatenated.
        """
        if self.measurement_type != "normal":
            logger.warning(f"Optics not available for {self.day} {self.tree} {self.measurement} {self.measurement_type}")
            return ""
        return f"{self.datapath}/parquet/{self.day.replace('-','_')}/{self.tree}_{self.measurement}_pulling.parquet"
    
    @property
    def file_acc_name(self):
        """
        Returns the file with ACC data. The file is 
        interpolated to the frequency 100Hz.
        Renames columns and adds time to index.
        """
        return f"{self.datapath}/parquet_acc/{self.measurement_type}_{self.day}_{self.tree}_{self.measurement}.parquet"

    @property
    def file_acc5000_name(self):
        """
        Returns the file with ACC data. The file is 
        sampled at 5000Hz.
        Renames columns and adds time to index.
        """
        return f"{self.datapath}/parquet_acc/{self.measurement_type}_{self.day}_{self.tree}_{self.measurement}_5000.parquet"

    @property
    def identify_major_minor(self):
        if self.measurement != "M01":
            return DynatreeMeasurement(
                self.day, self.tree, 
                "M01", measurement_type=self.measurement_type).identify_major_minor
        list_inclino = ["Inclino(80)", "Inclino(81)"]
        colors = {"Inclino(80)":"blue", "Inclino(81)":"yellow"}
        df = pd.read_parquet(self.file_pulling_name)
        df = fix_inclinometers_sign(
            df, self.measurement_type, self.day, self.tree)
        ans = {}
        for inclino in list_inclino:
            # najde maximum, major ma maximum kladne a vysoke
            maxima = df[[f"{inclino}X",f"{inclino}Y"]].max()
            # vytvori sloupce blue_Maj, blue_Min, yellow_Maj,  yellow_Min....hlavni a vedlejsi osa
            if maxima[f"{inclino}X"]>maxima[f"{inclino}Y"]:
                ans[f"{inclino}Maj"] = f"{inclino}X"
                ans[f"{inclino}Min"] = f"{inclino}Y"
                ans[f"{colors[inclino]}Maj"] = f"{inclino}X"
                ans[f"{colors[inclino]}Min"] = f"{inclino}Y"
            else:
                ans[f"{inclino}Maj"] = f"{inclino}Y"
                ans[f"{inclino}Min"] = f"{inclino}X"
                ans[f"{colors[inclino]}Maj"] = f"{inclino}Y"
                ans[f"{colors[inclino]}Min"] = f"{inclino}X"
        return ans

    @staticmethod
    def add_total_angle(df, major_dict):
        """
        The input is a dataframe with columns including Inclino(80)X, Inclino(80)Y, 
        Inclino(81)X, Inclino(81)Y and dictionary major_dict wihch may look like this:
        {'Inclino(80)Maj': 'Inclino(80)Y', 'Inclino(80)Min': 'Inclino(80)X', 
         'Inclino(81)Maj': 'Inclino(81)X', 'Inclino(81)Min': 'Inclino(81)Y'}
        The output is the dataframe extended by the columns with total angle. 
        The total angle preserves the sign of the major axis (thus the main 
        peak is positive). 
        """
        list_inclino = ["Inclino(80)", "Inclino(81)"]
        for inclino in list_inclino:
            major = major_dict[f"{inclino}Maj"]
            minor = major_dict[f"{inclino}Min"]

            if isinstance(df.columns, pd.MultiIndex):
                target = (inclino,"nan")
            else:
                target = inclino
            df.loc[:,target] = (arctand(
                np.sqrt((tand(df.loc[:,major]))**2 
                        + (tand(df.loc[:,minor]))**2 )
                ) * np.sign(df.loc[:,major])).values
        return df
    
    @cached_property
    def data_pulling(self):
        """
        Data from pulling test devices (extenzometer, inclinometer, 
        force gauge).
        
        Major axis is rescaled such that the main peak is positive.
        """
        logger.debug("loading pulling data")
        df = pd.read_parquet(self.file_pulling_name)
        df = fix_inclinometers_sign(df, self.measurement_type, self.day, self.tree)
        idx = (self.measurement_type,self.day, self.tree, self.measurement)
        if idx in df_reset_inclinometers.index:
            row = df_reset_inclinometers.loc[idx,:]
            for key, value in row.items():
                if not isinstance(value, float):
                    value = value.iloc[0]
                if not pd.isna(value):
                    shift = df.loc[value:,key].dropna().iloc[0]
                    df[key] = df[key] - shift
        df = DynatreeMeasurement.add_total_angle(df,self.identify_major_minor)
        return df
    
    @cached_property
    def data_acc(self):
        logger.debug("loading acc data")
        df = pd.read_parquet(self.file_acc_name)
        df.columns = [i.replace("Data1_","").replace("ACC","A0").replace("_axis","").lower() for i in df.columns]
        df = df.reindex(columns=sorted(df.columns))
        df.index = np.array(range(len(df.index)))/100
        return df 

    @cached_property
    def data_acc5000(self):
        logger.debug("loading acc data at 5000Hz")
        df = pd.read_parquet(self.file_acc5000_name)
        df.columns = [i.replace("Data1_","").replace("ACC","A0").replace("_axis","").lower() for i in df.columns]
        df = df.reindex(columns=sorted(df.columns))
        df.index = np.array(range(len(df.index)))/5000
        return df 

    @cached_property
    def data_pulling_interpolated(self):
        logger.debug("interpolating pulling data")
        df = self.data_pulling
        ans = df.interpolate(method='index')
        return  ans

    @cached_property
    def data_acc_interpolated(self):
        logger.debug("interpolating acc data")
        df = self.data_acc
        ans = df.interpolate(method='index')
        return  ans

    @cached_property
    def data_optics(self):
        logger.debug("loading optics data")
        ans = pd.read_parquet(self.file_optics_name)
        ans = ans[ans.index.notnull()]  # some rows at the end may have nan index
        return ans

    @cached_property
    def data_optics_pt34(self):
        logger.debug("loading optics data Pt3 and Pt4")
        ans = pd.read_parquet(
              self.file_optics_name, columns=["('Pt3', 'Y0')", "('Pt4', 'Y0')", "('Time', '')"])
        ans = ans[ans.index.notnull()]  # some rows at the end may have nan index
        return ans
        
    @cached_property
    def data_optics_extra(self):
        """
        Pulling data interpolated to the optics, synced with optics and with
        extra columns where we try to fix the movement of cameras.
        """
        logger.debug("loading optics extra")
        ans = pd.read_parquet(self.file_optics_extra_name)
        ans = ans[ans.index.notnull()]  # some rows at the end may have nan index
        # the sign is already fixed in the parquet_add_inclino.py
        #ans = fix_inclinometers_sign(ans, self.measurement_type, self.day, self.tree)
        # ans = DynatreeMeasurement.add_total_angle(ans,self.identify_major_minor)
        return ans

    @cached_property
    def data_optics_extra_reduced(self):
        """
        Like optics extra, but only the data from pulling device are considered.
        The column index converted from multiindex to plain index.
        """
        logger.debug("loading optics extra reduced")
        ans = self.data_optics_extra
        return ans

    @property
    def is_optics_available(self):
        return os.path.isfile(self.file_optics_name)

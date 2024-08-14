#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:25:19 2024

@author: marik
"""

DIRECTORY = "../data"
import pandas as pd
import glob
import os
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.interpolate import interp1d
from lib_dynatree import read_data_inclinometers, read_data_by_polars


def split_df_static_pulling(
        df_ori, 
        intervals=None,
        upper_bound=0.8,
        lower_bound=0.2,
        ):
    """
    Analyzes data in static tests, with three pulls. Inputs the dataframe, 
    outputs the list of dictionaries with minimal force, maximal force 
    df_ori is the dataframe as obtained from read_data_inclinometers
    """
    df= df_ori.copy()[["Force(100)"]].dropna()
    
    # Interpolace a vyhlazeni
    new_index = np.arange(df.index[0],df.index[-1],0.1)
    window_length = 100
    polyorder = 3
    newdf = df[["Force(100)"]].dropna()
    interpolation_function = interp1d(newdf.index, newdf["Force(100)"], kind='linear')
    df_interpolated = pd.DataFrame(interpolation_function(new_index), index=new_index, columns=["Force(100)"])
    df_interpolated['Force_smoothed'] = savgol_filter(df_interpolated['Force(100)'], window_length=window_length, polyorder=polyorder)

    steps_down=None
    if intervals is None:
        # Divide domain into interval where force is large and small
        maximum = df_interpolated["Force(100)"].max()
        df_interpolated["Force_step"] = (df_interpolated["Force(100)"]>0.5*maximum).astype(int) 

        # identify jumps down
        diff_d1 = df_interpolated["Force_step"].diff()
        steps_down = list(diff_d1.index[diff_d1<0])
        intervals = zip([0]+steps_down[:-1],steps_down)

    time = []
    for start,end in intervals:
        df_subset = df.loc[start:end,"Force(100)"]
        maximum_idx = np.argmax(df_subset)
        t = {'maximum': df_subset.index[maximum_idx]}
        # maximum_force = np.max(df_subset)
        # upper_limit_force_idx = np.argmax(df_subset>upper_bound*maximum_force)
        # upper_limit_force_time = df_subset.index[upper_limit_force_idx]
        # t['upper_limit_force'] = upper_limit_force_time
    
        df_subset = df.loc[start:t['maximum'],"Force(100)"]
        idxmin = df_subset.idxmin()
        t['minimum'] = idxmin
    
        # df_subset = df_subset[idxmin:]
        # df_subset = df_subset[::-1]
        
        # lower_limit_force_idx = np.argmax(df_subset<lower_bound*maximum_force)
        # lower_limit_force_time = df_subset.index[lower_limit_force_idx]
        # t['lower_limit_force'] = lower_limit_force_time
        time = time + [t]
    return {'times':time, 'df_interpolated': df_interpolated}

def find_intervals_to_split_measurements(date, tree, csv="csv/intervals_split_M01.csv"):
    df = pd.read_csv(csv, index_col=[], dtype={"tree":str}, sep=";")
    select = df[(df["date"]==date) & (df["tree"]==tree)]
    if select.shape[0] == 0:
        return None
    elif select.shape[0] > 1:
        print (f"Warning, multiple pairs of date-tree in file {csv}")
        select = select.iat[0,:]
    return np.array([int(i) for i in select["intervals"].values[0].replace("[","").replace("]","").split(",")]).reshape(-1,2)

#%%
measurement = "1"
date = "2021-03-22"
year,month,day=date.split("-")
tree = "16"

def get_static_pulling_data(year, month, day, tree, measurement, directory=DIRECTORY):
    if measurement == "1":
        df = read_data_inclinometers(f"{directory}/pulling_tests/{year}_{month}_{day}/BK_{tree}_M{measurement}.TXT")
        out = split_df_static_pulling(df, intervals = find_intervals_to_split_measurements(date, tree))
        times = out['times']
    else:
        df = read_data_by_polars(f"{directory}/csv_extended/{year}_{month}_{day}/BK{tree}_M0{measurement}.csv")
        times = [{"minimum":0, "maximum": df["Force(100)"].idxmax().values[0]}]
    return {'times': times, 'dataframe': df}


out = get_static_pulling_data(year, month, day, tree, measurement)
out = get_static_pulling_data(year, month, day, tree, "3")
out = get_static_pulling_data(year, month, day, tree, "2")

#%%
times = out['times']
dataframe = out['dataframe']
ax = dataframe["Force(100)"].plot()
ax.set(title=f"{year}-{month}-{day} BK{tree} M0{measurement}")
for _ in times:
    dataframe.loc[_['minimum']:_['maximum'],"Force(100)"].plot(ax=ax)
    
# #%%
# %time
# measurement = "2"
# df = read_data_by_polars(f"{DIRECTORY}/csv_extended/{year}_{month}_{day}/BK{tree}_M0{measurement}.csv")
# df.index.name="Time"
# {"minimum":0, "maximum": df["Force(100)"].idxmax()}

 
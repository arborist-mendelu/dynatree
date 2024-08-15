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
    df_ori is the dataframe as obtained from read_data_inclinometers.
    
    Initial estimate of subintervals can be provided. If not, it is detected 
    automatically.
    
    
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


def process_inclinometers_major_minor(df_, height_of_anchorage=None, rope_angle=None, 
                          height_of_pt=None):
    """
    The input is dataframe loaded by pd.read_csv from pull_tests directory.
    
    * Converts Inclino(80) and Inclino(81) to blue and yellow respectively.
    * Evaluates the total angle of inclination
    * Evaluates horizontal and vertical forces
    * 
    """
    df = pd.DataFrame(index=df_.index, columns=["blue","yellow"])
    df[["blueX","blueY","yellowX","yellowY"]] = df_[["Inclino(80)X","Inclino(80)Y", "Inclino(81)X","Inclino(81)Y",]]
    for inclino in ["blue","yellow"]:
        df.loc[:,[inclino]] = arctand(np.sqrt((tand(df[f"{inclino}X"]))**2 + (tand(df[f"{inclino}Y"]))**2 ))
        # najde maximum bez ohledu na znamenko
        maxima = df[[f"{inclino}X",f"{inclino}Y"]].abs().max()
        # vytvori sloupce blue_Maj, blue_Min, yellow_Maj,  yellow_Min....hlavni a vedlejsi osa
        if maxima[f"{inclino}X"]>maxima[f"{inclino}Y"]:
            df.loc[:,[f"{inclino}_Maj"]] = df[f"{inclino}X"]
            df.loc[:,[f"{inclino}_Min"]] = df[f"{inclino}Y"]
        else:
            df.loc[:,[f"{inclino}_Maj"]] = df[f"{inclino}Y"]
            df.loc[:,[f"{inclino}_Min"]] = df[f"{inclino}X"]
        # Najde pozici, kde je extremalni hodnota - nejkladnejsi nebo nejzapornejsi
        idx = df[f"{inclino}_Maj"].abs().idxmax()
        # promenna bude jednicka pokus je extremalni hodnota kladna a minus
        # jednicka, pokud je extremalni hodnota zaporna
        if pd.isna(idx):
            znamenko = 1
        else:
            znamenko = np.sign(df[f"{inclino}_Maj"][idx])
        # V zavisosti na znamenku se neudela nic nebo zmeni znamenko ve sloupcich
        # blueM, blueV, yellowM, yellowV
        for axis in ["_Maj", "_Min"]:
            df.loc[:,[f"{inclino}{axis}"]] = znamenko * df[f"{inclino}{axis}"]    
        # convert index to multiindex
        return df

def process_forces(df_, height_of_anchorage=None, rope_angle=None, 
                          height_of_pt=None):
    """
    Input is a dataframe with Force(100) column
    """
    df = pd.DataFrame(index=df_.index)
    # evaluate the horizontal and vertical component
    if rope_angle is None:
        # If rope angle is not given, use the data from the table
        rope_angle = df_['RopeAngle(100)']
    # evaluate horizontal and vertical force components and moment
    df.loc[:,['F_horizontal']] = df_['Force(100)'] * np.cos(np.deg2rad(rope_angle))
    df.loc[:,['F_vertical']] = df_['Force(100)'] * np.sin(np.deg2rad(rope_angle))
    df.loc[:,['M']] = df['F_horizontal'] * height_of_anchorage
    df.loc[:,['M_Pt']] = df['F_horizontal'] * ( height_of_anchorage - height_of_pt )
    sloupce = [i.split("/") for i in df.columns]
    sloupce = [i if len(i)>1 else [i[0],'nan'] for i in sloupce ]
    # df.columns = pd.MultiIndex.from_tuples(sloupce)
    return df
    
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

#%%
# measurement = "1"
# day = "2021-03-22"
# tree = "16"


df_pt_notes = pd.read_csv("csv/PT_notes_with_pt.csv", sep=",")
df_pt_notes.index = df_pt_notes["tree"]

def get_static_pulling_data(day, tree, measurement, directory=DIRECTORY):
    measurement = measurement[-1]
    tree = tree[-2:]
    print(tree, measurement)
    if measurement == "1":
        df = read_data_inclinometers(f"{directory}/pulling_tests/{day.replace('-','_')}/BK_{tree}_M{measurement}.TXT")
        out = split_df_static_pulling(df, intervals = find_intervals_to_split_measurements(day, tree))
        times = out['times']
    else:
        df = read_data_by_polars(f"{directory}/csv_extended/{day.replace('-','_')}/BK{tree}_M0{measurement}.csv")
        times = [{"minimum":0, "maximum": df["Force(100)"].idxmax().values[0]}]
    return {'times': times, 'dataframe': df}



#%%

def nakresli(day, tree, measurement):
    out = get_static_pulling_data(day, tree, measurement)
    out['dataframe']=out['dataframe'].interpolate()
    tree = tree[-2:]
    measurement = measurement[-1]
    df_with_major = process_inclinometers_major_minor(out['dataframe'])

    df_with_forces = process_forces(
        out['dataframe'], 
        height_of_anchorage=df_pt_notes.at[int(tree),'height_of_anchorage'], 
        height_of_pt=df_pt_notes.at[int(tree),'height_of_pt']
        )

    # Rope angle
    # rope_angle = df_pt_notes.at[int(tree),'angle_of_anchorage']

    #%%
    times = out['times']
    dataframe = out['dataframe']
    fig, ax = plt.subplots()
    dataframe["Force(100)"].plot(ax=ax)
    ax.set(title=f"{day} BK{tree} M0{measurement}")
    for _ in times:
        dataframe.loc[_['minimum']:_['maximum'],"Force(100)"].plot(ax=ax)
    return fig
    
#%%

# dataframe.loc[_['minimum']:_['maximum'],:]

 
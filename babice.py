#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% Functions and procedures
"""
Created on Thu Nov  2 14:09:25 2023

@author: marik
"""

import os
PATH = "/mnt/ERC/ERC/"
PATH = "../"

# MEASUREMENT_DAY =  "01_Mereni_Babice_22032021_optika_zpracovani"   
# MEASUREMENT_DAY =  "01_Mereni_Babice_29062021_optika_zpracovani"   
# MEASUREMENT_DAY =  "01_Mereni_Babice_05042022_optika_zpracovani"   
MEASUREMENT_DAY =  "01_Mereni_Babice_16082022_optika_zpracovani"   


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from scipy import interpolate
import re

from lib_dynatree import read_data
from lib_dynatree import directory2date
from lib_dynatree import filename2tree_and_measurement_numbers

def fix_data_by_points_on_ground(df):
    """
    Parameters
    ----------
    df : Dataframe to be fixed

    Returns
    -------
    Dataframe with pair Pt0, Pt1 or Pt3, Pt4 fixed by subtracting the coordiantes 
    of Pt8, Pt9, Pt10 (CAM1) and Pt11, Pt12, Pt13 (Cam0)
    Adds columns like Pt0_fixed_by_8. 
    All numbers remain intact.
    """
    for by in range(8,14):
        if by in [8, 9, 10]:
            targets = [0, 1]
        elif by in [11, 12, 13]:
            targets = [3, 4]
        else:
            warnings.warn("The value of by should be one of 8,9,10,11,12,13. The dataframe is kept intact.")
            return df
            
        """
        Calculate the fix. 
        """    
        fix = df.xs(key=f"Pt{by}", level=0, axis=1)
        fix = fix - fix.iloc[0,:]    
        for target in targets:
            for i in ['X0','Y0']:
                df[(f"Pt{target}_fixed_by_{by}",i)]  = df[(f"Pt{target}",i)] - fix[i]
    return df
    

def plot_points_on_ground(df, figsize=(15,10), suptitle=""):
    """
    Plots points on ground. The points do mot move (unless the ground is moved).
    Used to fix the camera movement etc.
    
    Parameters
    ----------
    df : dataframe

    Returns
    -------
    Plots two figures. Returns a list of two figures. These figures can be saved.

    """
    figs, axs = plt.subplots(2,2,figsize=figsize)
    for i_a, nehybne_body in enumerate([["8", "9", "10"],["11", "12", "13"]]):
        ax = axs[i_a,:]
        for i in nehybne_body:
            for j,axis in enumerate(["X0","Y0"]):
                ax[j].plot(df["Time"], df[(f"Pt{i}",axis)] - df.loc[0,(f"Pt{i}",axis)], label=f"Pt{i} - {axis}", alpha=0.5)
    for ax in axs.reshape(-1):
        ax.legend()      
    plt.suptitle(suptitle)
    plt.tight_layout()
    return figs

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

def read_data_inclinometers(file, release=None):
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
    df_pulling_tests.interpolate(inplace=True)
    if release is None:
        return df_pulling_tests
    
    # Find the release time. Estimated as maximum of sum squares of
    # values of inclinometers. The first 10 seconds are skipped, in
    # some measurement some strange large values are at the initial
    # phase.
    
    i1_data = df_pulling_tests["Inclino(80)X"]**2 + df_pulling_tests["Inclino(80)Y"]**2
    i2_data = df_pulling_tests["Inclino(81)X"]**2 + df_pulling_tests["Inclino(81)Y"]**2
    i1_data = i1_data[10:]
    i2_data = i2_data[10:]
    
    i1_release_time = i1_data.idxmax()
    i2_release_time = i2_data.idxmax()
    if i1_data[i1_release_time] > i2_data[i2_release_time]:
        release_time_inclino = i1_release_time
    else:
        release_time_inclino = i2_release_time
        
    # Sync the dataframe from inclino to optics    
    df_pulling_tests["Time_inclino"] = df_pulling_tests.index
    df_pulling_tests["Time"] = df_pulling_tests["Time_inclino"] - release_time_inclino + release
    df_pulling_tests.set_index("Time", inplace=True)
    df_pulling_tests["Time"] = df_pulling_tests.index
    
    #        [m1,i1] = max(T.Inclino_80_X(100:end).^2 + T.Inclino_80_Y(100:end).^2);
    #        [m2,i2] = max(T.Inclino_81_X(100:end).^2 + T.Inclino_81_Y(100:end).^2);
        
    return df_pulling_tests

def add_data_from_inclinometers(df, df_pulling_tests):
    cols = [i for i in df_pulling_tests.columns if "Inclino" in i or "Force" in i or "Elasto" in i or "Rope" in i]
    for col in cols:
        f = interpolate.interp1d(df_pulling_tests.index, df_pulling_tests[col], axis=0, fill_value="extrapolate")
        df[col] = f(df.index)
    return df

def extend_csv(measurement_day, path="../"):
    """
    Reads csv files in a csv directory, adds data from inclinometers 
    and saves to csv_extendd directory
    

    Returns
    -------
    None.

    """
    files = os.listdir(f"{path}{measurement_day}/csv/")
    files.sort()
    for f in files:
        tree,tree_measurement = filename2tree_and_measurement_numbers(f)
        print (tree, tree_measurement)
    
        # Read data file
        # In spyder run cell with Ctrl+Enter
            
        # načte data z csv souboru
        df = read_data(f"{path}{measurement_day}/csv/BK{tree}_M0{tree_measurement}.csv")   
        # prida sloupce s odectenim pohybu bodu na zemi
        df_fixed = df.copy().pipe(fix_data_by_points_on_ground) 
        # přidá data z inklinoměrů, synchronizuje a interpoluje na stejné časové okamžiky
        release_time_optics = find_release_time_optics(df)
        df_pulling_tests = read_data_inclinometers(
            f"{path}{measurement_day}/pulling_tests/BK_{tree}_M{tree_measurement}.TXT", 
            release=release_time_optics
            )
        df_fixed_and_inclino = df_fixed.copy().pipe(add_data_from_inclinometers, df_pulling_tests)
        df_fixed_and_inclino.to_csv(f"{path}{measurement_day}/csv_extended/BK{tree}_M0{tree_measurement}.csv")


df_remarks = pd.read_csv("csv/oscillation_times_remarks.csv")
# %% Create subdirectories

for d in ["csv_extended", "png_points_on_ground", "png_with_inclino"]:
    try:
       os.makedirs(f"{PATH}{MEASUREMENT_DAY}/{d}")
    except FileExistsError:
       # directory already exists
       pass
# %% Extend csv files in csv directory to csv_rextended with inclino data
# Extend csv files in csv directory to csv_extended with inclino data 

extend_csv(MEASUREMENT_DAY)


# %% Pre-release data, also plot fixed pt3 and plot data from inlinometers, 

measurement_day = MEASUREMENT_DAY

files = os.listdir(f"{PATH}{measurement_day}/csv_extended/")
files.sort()
pre_release_data = {}
files = [""]

for file in files[:]:
    print(file)    
    tree,tree_measurement = filename2tree_and_measurement_numbers(file)
    bounds_for_fft = df_remarks[(df_remarks["tree"]==f"BK{tree}") & (df_remarks["measurement"]==f"M0{tree_measurement}") & (df_remarks["date"]==directory2date(measurement_day))]
 
    df = read_data(f"{PATH}{measurement_day}/csv_extended/{file}", index_col=0)
    fix_target = 3
    plot_coordiante = "Y"
    
    
    fixes = [i for i in df.columns if f"{fix_target}_fixed_by" in i[0] and plot_coordiante in i[1]]
    
    fig, axes = plt.subplots(3,1,figsize=(12,8),sharex=True)
    plt.suptitle(f"{measurement_day.replace('_optika_zpracovani','')} - BK{tree} M0{tree_measurement}")

    ax = axes[0]
    df[ [(f'Pt{fix_target}', f'{plot_coordiante}0')]+ fixes ].plot(ax=ax)
    ax.legend(title="",loc=2)
    ax.set(title=f"Pt{fix_target} and fixes based on points on ground")    
    ax.grid()
    t = ax.text(
        0, 0, 
        bounds_for_fft['remark'].values[0],
        ha='left', 
        va='bottom',
        transform=ax.transAxes,
        color="r",
        backgroundcolor="white",
        wrap=True)
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))

    release_time_optics = find_release_time_optics(df)
    start = release_time_optics-5
    start = 0
    ax = axes[1]    
    # ax = [axs, axs.twinx()]       
    # df.loc[start:,[("Pt3","Y0")]].plot(ax=ax[0])
    list_inclino = ["Inclino(80)X","Inclino(80)Y","Inclino(81)X","Inclino(81)Y"]
    df.loc[start:,list_inclino].plot(ax=ax)
    ax.grid()
    ax.legend(list_inclino, title="")
    ax.set(title="Inclinometers")
    
    ax = axes[2]
    df.loc[start:,"Force(100)"].plot(ax=ax)
    ax.grid()
    lines, labels = ax.get_legend_handles_labels()
    ax.legend().remove()

    ax = ax.twinx()
    df.loc[start:,"Elasto(90)"].plot(ax=ax,color="C1")
    ax.set(title="Force and Elasto")
    lines2, labels2 = ax.get_legend_handles_labels()
    ax.legend(lines + lines2, ["Force","Elasto"])
    ax.grid(color="C1")
    ax.tick_params(axis='y', labelcolor="C1")     
    
    fig.tight_layout()
    
    maxforceidx = df["Force(100)"].idxmax().values[0]
    maxforce  = df["Force(100)"].max().values[0]
    percent1 = 0.95
    tmax = np.abs(df.loc[:maxforceidx,["Force(100)"]]-maxforce*percent1).idxmin().values[0]
    percent2 = 0.85
    tmin = np.abs(df.loc[:maxforceidx,["Force(100)"]]-maxforce*percent2).idxmin().values[0]

    sloupce = ["Time"]+list_inclino+["Force(100)","Elasto(90)","Pt3","Pt4"]
    delta_df = df[sloupce].copy()
    for i in ["Pt3", "Pt4"]:
        delta_df[i] = delta_df[i] - delta_df[i].iloc[0]
    delta_df = delta_df.loc[tmin:tmax,:]    
    # ensure that Force info is not
    if not pd.isna(tmin):
        time_middle = (tmin+tmax)/2
        time_middle = df.loc[time_middle:,"Time"].iloc[0].values[0]
        for ax in axes:
            ax.axvspan(tmin,tmax, alpha=.5, color="yellow")
            pre_release_data[file.replace(".csv","")] = delta_df.mean()
    else:
        #release_data[file.replace(".csv","")] = []
        pass
    lower_bound = bounds_for_fft["start"].values[0]
    upper_bound = bounds_for_fft["end"].values[0] 
    if upper_bound == np.inf:
        upper_bound = df["Time"].max().values[0]
    axes[0].axvspan(lower_bound, upper_bound, alpha=0.5, color="gray")
    # break
    fig.savefig(f"{PATH}{measurement_day}/png_with_inclino/{file.replace('csv','png')}")
    plt.close('all')
    

# %% Save pre_release data

key = list(pre_release_data.keys())[0]
columns = [ i[0]+re.sub("Unnam.*","",i[1]) for i in pre_release_data[key].index]
columns[-4:] = ["PT3_UX0", "PT3_UY0", "PT4_UX0", "PT4_UY0"]


for key in pre_release_data.keys():
    pre_release_data[key].index = columns
pre_release_df = pd.DataFrame(pre_release_data).T
pre_release_df.to_excel(f"pre_release_data_{MEASUREMENT_DAY}.xlsx")
pre_release_df

# %% Plot probes on ground

measurement_day = MEASUREMENT_DAY

files = os.listdir(f"{PATH}{measurement_day}/csv_extended/")
files.sort()    
for file in files:
    df = read_data(f"{PATH}{MEASUREMENT_DAY}/csv_extended/{file}", index_col=0)
    figs = plot_points_on_ground(df, suptitle=f"{MEASUREMENT_DAY} - {file}")
    figs.savefig(f"{PATH}{MEASUREMENT_DAY}/png_points_on_ground/{file.replace('.csv','.png')}")
    plt.close('all')

# %%
# df = read_data(f"{MEASUREMENT_DAY}/csv_extended/BK01_M02.csv")

# df = df["Force(100)"]
# df.plot()
# %%


# %%

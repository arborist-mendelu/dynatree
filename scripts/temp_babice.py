#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% Functions and procedures
"""
Created on Thu Nov  2 14:09:25 2023

@author: DYNATREE project, ERC CZ no. LL1909 "Tree Dynamics: Understanding of Mechanical Response to Loading"
"""

# PATH = "/mnt/ERC/ERC/"
# PATH = "../"

# MEASUREMENT_DAY =  "01_Mereni_Babice_22032021_optika_zpracovani"   
# MEASUREMENT_DAY =  "01_Mereni_Babice_29062021_optika_zpracovani"   
# MEASUREMENT_DAY =  "01_Mereni_Babice_05042022_optika_zpracovani"   
# MEASUREMENT_DAY =  "01_Mereni_Babice_16082022_optika_zpracovani"   


import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import re

from dynatree.dynatree import read_data
from dynatree.dynatree import directory2date
from dynatree.dynatree import find_release_time_optics
from dynatree.dynatree import filename2tree_and_measurement_numbers


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


# df_remarks = pd.read_csv("csv/oscillation_times_remarks.csv")
# %% Create subdirectories

for d in ["png_points_on_ground", "png_with_inclino"]:
    try:
       os.makedirs(f"{PATH}{MEASUREMENT_DAY}/{d}")
    except FileExistsError:
       # directory already exists
       pass

# %% Pre-release data, also plot fixed pt3 and plot data from inlinometers, 

measurement_day = MEASUREMENT_DAY

files = os.listdir(f"{PATH}{measurement_day}/csv_extended/")
files.sort()
pre_release_data = {}
# files = ["BK11_M03.csv"]

for file in files[:]:
    print(file)    
    tree,tree_measurement = filename2tree_and_measurement_numbers(file)
    bounds_for_fft = df_remarks[(df_remarks["tree"]==f"BK{tree}") & (df_remarks["measurement"]==f"M0{tree_measurement}") & (df_remarks["date"]==directory2date(measurement_day))]
 
    df = read_data(f"{PATH}{measurement_day}/csv_extended/{file}", index_col=0)
    fix_target = 3
    plot_coordiante = "Y"
    
    
    fixes = [i for i in df.columns if f"{fix_target}_fixed_by" in i[0] and plot_coordiante in i[1]]
    
    fig, axes = plt.subplots(3,1,figsize=(11.3,8),sharex=True)
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
        # time_middle = (tmin+tmax)/2
        # time_middle = df.loc[time_middle:,"Time"].iloc[0].values[0]
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

# measurement_day = MEASUREMENT_DAY

# files = os.listdir(f"{PATH}{measurement_day}/csv_extended/")
# files.sort()    
# for file in files:
#     df = read_data(f"{PATH}{MEASUREMENT_DAY}/csv_extended/{file}", index_col=0)
#     figs = plot_points_on_ground(df, suptitle=f"{MEASUREMENT_DAY} - {file}")
#     figs.savefig(f"{PATH}{MEASUREMENT_DAY}/png_points_on_ground/{file.replace('.csv','.png')}")
#     plt.close('all')

# %%
# df = read_data(f"{MEASUREMENT_DAY}/csv_extended/BK01_M02.csv")

# df = df["Force(100)"]
# df.plot()
# %%


# %%

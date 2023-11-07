#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:07:36 2023

@author: marik
"""

import os
import glob
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
import warnings
from scipy import interpolate
import re

from lib_dynatree import read_data
from lib_dynatree import directory2date
from lib_dynatree import find_release_time_optics
from lib_dynatree import filename2tree_and_measurement_numbers

# measurement_day="01_Mereni_Babice_22032021_optika_zpracovani"
# path="../"
# tree="01"
# tree_measurement="2"

df_remarks = pd.read_csv("csv/oscillation_times_remarks.csv")


def plot_one_measurement(measurement_day="01_Mereni_Babice_22032021_optika_zpracovani", path="../", tree="01", tree_measurement="2", df_remarks=df_remarks):
    df = read_data(f"{path}{measurement_day}/csv/BK{tree}_M0{tree_measurement}.csv")   
    df_extra = read_data(f"{path}{measurement_day}/csv_extended/BK{tree}_M0{tree_measurement}.csv")   
    bounds_for_fft = df_remarks[(df_remarks["tree"]==f"BK{tree}") & (df_remarks["measurement"]==f"M0{tree_measurement}") & (df_remarks["date"]==directory2date(measurement_day))]
    fix_target = 3
    plot_coordiante = "Y"
    fixes = [i for i in df_extra.columns if f"{fix_target}_fixed_by" in i[0] and plot_coordiante in i[1]]
    
    fig, axes = plt.subplots(3,1,figsize=(11.3,8),sharex=True)
    plt.suptitle(f"{measurement_day.replace('_optika_zpracovani','')} - BK{tree} M0{tree_measurement}")
    
    # Plot probes, region of interest for oscillation
    ax = axes[0]
    df_extra[fixes].plot(ax=ax)
    df[(f'Pt{fix_target}', f'{plot_coordiante}0')].plot(ax=ax)
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
    lower_bound = bounds_for_fft["start"].values[0]
    upper_bound = bounds_for_fft["end"].values[0] 
    if upper_bound == np.inf:
        upper_bound = df["Time"].max()
    ax.axvspan(lower_bound, upper_bound, alpha=0.5, color="gray")
        
    # plot inclinometers
    release_time_optics = find_release_time_optics(df)
    start = release_time_optics-5
    start = 0
    ax = axes[1]    
    list_inclino = ["Inclino(80)X","Inclino(80)Y","Inclino(81)X","Inclino(81)Y"]
    df_extra.loc[start:,list_inclino].plot(ax=ax)
    ax.grid()
    ax.legend(list_inclino, title="", loc=3)
    ax.set(title="Inclinometers")
        
    # plot force and strain
    ax = axes[2]
    df_extra.loc[start:,"Force(100)"].plot(ax=ax)
    ax.grid()
    lines, labels = ax.get_legend_handles_labels()
    ax.legend().remove()
    ax = ax.twinx()
    df_extra.loc[start:,"Elasto(90)"].plot(ax=ax,color="C1")
    ax.set(title="Force and Elasto")
    lines2, labels2 = ax.get_legend_handles_labels()
    ax.legend(lines + lines2, ["Force","Elasto"], loc=3)
    ax.grid(color="C1")
    ax.tick_params(axis='y', labelcolor="C1")     
    
    fig.tight_layout()
    
    # Replace negative force by 0
    # df_extra.loc[df_extra["Force(100)"]<0,"Force(100)"] = 0
    # Find time for 95% and 85% force
    maxforceidx = df_extra["Force(100)"].idxmax().iat[0]
    maxforce  = df_extra["Force(100)"].max().iat[0]
    percent1 = 0.95
    tmax = np.abs(df_extra.loc[:maxforceidx,["Force(100)"]]-maxforce*percent1).idxmin().values[0]
    percent2 = 0.85
    tmin = np.abs(df_extra.loc[:maxforceidx,["Force(100)"]]-maxforce*percent2).idxmin().values[0]
    
    sloupce = ["Time"]+list_inclino+["Force(100)","Elasto(90)"]
    delta_df_extra = df_extra[sloupce].copy()
    delta_df = df[["Pt3","Pt4"]].copy()
    for i in ["Pt3", "Pt4"]:
        delta_df[i] = delta_df[i] - delta_df[i].iloc[0]
    delta_df = delta_df.loc[tmin:tmax,:]    
    delta_df_extra = delta_df_extra.loc[tmin:tmax,:]    
    for ax in axes:
        ax.axvspan(tmin,tmax, alpha=.5, color="yellow")
        # pre_release_data[file.replace(".csv","")] = delta_df.mean()
    fig.savefig(f"{path}{measurement_day}/png_with_inclino/BK{tree}_M0{tree_measurement}.png")
    plt.close('all')

def plot_one_day(measurement_day="01_Mereni_Babice_22032021_optika_zpracovani", path="../", df_remarks=df_remarks):
    
    csvfiles =  glob.glob(f"../{measurement_day}/csv/*.csv")
    csvfiles.sort()
    for file in csvfiles:
        filename = file.split("/")[-1]
        print(filename,", ",end="")
        tree = filename[2:4]
        tree_measurement = filename[7]
        plot_one_measurement(measurement_day=measurement_day, path=path, tree=tree, tree_measurement=tree_measurement, df_remarks=df_remarks)
    print(f"Konec zpracování pro {measurement_day}")
    
def main():
    for i in [
                    "01_Mereni_Babice_22032021_optika_zpracovani",   
                    "01_Mereni_Babice_29062021_optika_zpracovani", 
                    "01_Mereni_Babice_05042022_optika_zpracovani",
                    "01_Mereni_Babice_16082022_optika_zpracovani",
                    ]:
        print(i)
        print("=====================================================")
        plot_one_day(measurement_day=i)

main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:18:50 2023

Cte csv soubory. Hledá časový interval, kdy je síla v zadaných mezích a na 
tomto časovém intervalu vypočíta průměrnou hodnotu pro sílu, inklinometry, 
elastometr, delta Pt3 a delta Pt4.

@author: marik
"""

import os
import glob

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


def find_release_data_one_measurement(
        measurement_day="01_Mereni_Babice_22032021_optika_zpracovani",
        path="../", 
        tree="01", 
        tree_measurement="2", 
        ):
    # print("/nacitam soubory/", flush=True)
    df_main = read_data(
        f"{path}{measurement_day}/csv/BK{tree}_M0{tree_measurement}.csv")
    df_extra = read_data(
        f"{path}{measurement_day}/csv_extended/BK{tree}_M0{tree_measurement}.csv")

    # print("/extrahuji data/", flush=True)
    list_inclino = ["Inclino(80)X","Inclino(80)Y","Inclino(81)X","Inclino(81)Y"]
    df = pd.concat(
        [df_main[["Time","Pt3","Pt4"]],
         df_extra[["Force(100)", "Elasto(90)"]+list_inclino]
         ], axis=1)
    df = df - df.iloc[0,:]

    # print("/hledam casovy interval/")    
    if df["Force(100)"].isna().values.all():
        tmin = 0
        tmax = 0
    else:
        maxforceidx = df["Force(100)"].idxmax().values[0]
        maxforce  = df["Force(100)"].max().values[0]
        percent1 = 0.95
        tmax = np.abs(df.loc[:maxforceidx,["Force(100)"]]-maxforce*percent1).idxmin().values[0]
        percent2 = 0.85
        tmin = np.abs(df.loc[:maxforceidx,["Force(100)"]]-maxforce*percent2).idxmin().values[0]
    # Výběr časového intervalu
    df = df.loc[tmin:tmax,:]
    # Výpočet průměrů
    df_means = df.mean(skipna=True)
    # Převod z hierarchického indexu na flat index, odstranění nepotřebných dat
    df_means.drop([("Pt3","X0"),("Pt4","X0"),("Time",'')], inplace=True)
    df_means.index = [i[0] for i in df_means.index.to_flat_index()]
    return df_means

def find_release_data_one_day(measurement_day="01_Mereni_Babice_22032021_optika_zpracovani", path="../"):
    output = {}
    # Find csv files
    csvfiles =  glob.glob(f"../{measurement_day}/csv/*.csv")
    csvfiles.sort()
    # Drop directory name
    csvfiles = [i.split("/")[-1] for i in csvfiles]
    for file in csvfiles:
        tree, tree_measurement = filename2tree_and_measurement_numbers(file)
        print (f"BK{tree} M0{tree_measurement}, ",end="")
        output[f"BK{tree} M0{tree_measurement}"] = find_release_data_one_measurement(measurement_day=measurement_day, tree=tree, tree_measurement=tree_measurement, path=path)
    df = pd.DataFrame(output)
    return df    

def main():
    dfs = {}
    for i in [
                    "01_Mereni_Babice_22032021_optika_zpracovani",   
                    "01_Mereni_Babice_29062021_optika_zpracovani", 
                    "01_Mereni_Babice_05042022_optika_zpracovani",
                    "01_Mereni_Babice_16082022_optika_zpracovani",
                    ]:
        print()
        print(i)
        print("=====================================================")
        dfs[directory2date(i)] = find_release_data_one_day(measurement_day=i).T
        pd.DataFrame(dfs[directory2date(i)]).to_csv(f"release_data_{directory2date(i)}.csv")
    return dfs


if __name__ == "__main__":
    output = main()
    


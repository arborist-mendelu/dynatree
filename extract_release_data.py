#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 14:18:50 2023

extract_release_data.py
-----------------------
Cte csv soubory. Hledá časový interval, kdy je síla mezi 85 a 95 procenty maxima
(zhruba okamzik pred vypustenim) a na tomto časovém intervalu vypočíta průměrnou 
hodnotu pro sílu, inklinometry, elastometr, delta Pt3 a delta Pt4.
Prida hodnoty inklinomeru na zacatku mereni. Pokud jsou tyto hodnoty velke, asi 
nebyl vynulovany inklinometr nebo behem pocatecni faze "poskocil".

@author: marik
"""

import glob
import numpy as np
import pandas as pd

from lib_dynatree import read_data, read_data_selected
from lib_dynatree import directory2date, find_release_time_interval
from lib_dynatree import filename2tree_and_measurement_numbers


def find_release_data_one_measurement(
        date="2021-03-22",
        path="../data",
        tree="01",
        measurement="2",
        ):
    if len(tree)>2:
        tree = tree[-2:]
    if len(measurement)>1:
        measurement = measurement[-1]

    df_main = read_data_selected(
        f"{path}/parquet/{date.replace('-','_')}/BK{tree}_M0{measurement}.parquet")
    df_extra = read_data(
        f"{path}/parquet/{date.replace('-','_')}/BK{tree}_M0{measurement}_pulling.parquet")

    # print("/extrahuji data/", flush=True)
    list_inclino = ["Inclino(80)X","Inclino(80)Y","Inclino(81)X","Inclino(81)Y"]
    df = pd.concat(
        [df_main[["Time","Pt3","Pt4"]],
         df_extra[["Force(100)", "Elasto(90)","RopeAngle(100)"]+list_inclino]
         ], axis=1)
    df = df - df.iloc[0,:]

    # print("/hledam casovy interval/")
    tmin, tmax = find_release_time_interval(df_extra, date, tree, measurement)

    # Výběr časového intervalu
    df_release = df.loc[tmin:tmax,:].copy()
    # Výpočet průměrů
    df_means = df_release.mean(skipna=True)
    # Převod z hierarchického indexu na flat index, odstranění nepotřebných dat
    df_means.drop([("Pt3","X0"),("Pt4","X0"),("Time",'')], inplace=True)
    df_means.index = [i[0] for i in df_means.index.to_flat_index()]
    df_notna = df[df.index.notna()]
    for lb,ub in [[0,5],[5,10],[10,20]]:
        for inclino in list_inclino:
            df_means[f"{inclino}_{lb}_{ub}"]=np.mean(df_notna.loc[lb:ub,inclino])
    return df_means

def find_release_data_one_day(date="2021-03-22", path="../data"):
    output = {}
    # Find csv files
    files =  glob.glob(f"{path}/parquet/{date.replace('-','_')}/*.parquet")
    files.sort()
    # Drop directory name
    files = [i.split("/")[-1] for i in files]
    for file in files:
        tree, measurement = filename2tree_and_measurement_numbers(file)
        print (f"BK{tree} M0{measurement}, ",end="", flush=True)
        output[f"BK{tree} M0{measurement}"] = find_release_data_one_measurement(date=date, tree=tree, measurement=measurement, path=path)
    df = pd.DataFrame(output)
    return df

def main():
    dfs = {}
    for i in [ 
                    "2021-03-22",   
                    "2021-06-29", 
                    "2022-04-05",
                    "2022-08-16",
                    ]:
        print()
        print(i)
        print("=====================================================")
        dfs[i] = find_release_data_one_day(date=i).T
        pd.DataFrame(dfs[i]).to_csv(f"csv/release_data_{i}.csv")
    return dfs


if __name__ == "__main__":
    output = main()


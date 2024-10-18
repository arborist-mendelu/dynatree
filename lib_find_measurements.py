#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 17:33:44 2024

Finds measurements and collect into a dataframe.

@author: marik
"""

import glob
import pandas as pd
import numpy as np

DIRECTORY = "../data"

def get_all_measurements_pulling(cesta=DIRECTORY, suffix='parquet', directory='parquet_pulling'):
    """
    Gets all static measurements. Makes use of data from pulling experiments.
    """
    files = glob.glob(cesta+f"/{directory}/*/*.{suffix}") 
    files = [i.replace(cesta+f"/{directory}/","").replace(f".{suffix}","") for i in files]
    s = pd.Series(files)
    split = s.str.split('/', expand=True)
    info = split.iloc[:,1].str.split('_', expand=True)
    ans = {
        'date': split.iloc[:,0].str.replace("_","-"),
        'tree': info.iloc[:,1],
        'measurement': info.iloc[:,2],
        'type': info.iloc[:,0],
        }
    df = pd.DataFrame(ans).sort_values(by = ["date","tree","measurement"])
    df = df.reset_index(drop=True)
    return df

def get_all_measurements_acc(cesta=DIRECTORY, suffix='parquet', directory='parquet_acc'):
    """
    Gets all static measurements. Makes use of data from pulling experiments.
    """
    files = glob.glob(cesta + f"/{directory}/*.{suffix}")
    files = [i.replace(cesta + f"/{directory}/", "").replace(f".{suffix}", "").replace("_5000", "")
             for i in files]
    s = pd.Series(files)
    info = s.str.split('_', expand=True)
    ans = {
        'date': info.iloc[:,1].str.replace("_", "-"),
        'tree': info.iloc[:, 2],
        'measurement': info.iloc[:, 3],
        'type': info.iloc[:, 0],
    }
    df = pd.DataFrame(ans).sort_values(by=["date", "tree", "measurement"])
    df = df.reset_index(drop=True)
    return df

def get_all_measurements_optics(cesta=DIRECTORY, suffix='parquet', directory='parquet'):
    """
    Gets all measurements with optics.
    """
    files = glob.glob(cesta+f"/{directory}/*/*.{suffix}") 
    files = [i.replace(cesta+f"/{directory}/","").replace(f".{suffix}","") for i in files]
    df = pd.DataFrame(files, columns=["full_path"])
    df[["date","rest"]] = df['full_path'].str.split("/", expand=True)
    df["date"] = df["date"].str.replace("_","-")
    df["is_pulling"] = df["rest"].str.contains("pulling")
    mask = df["is_pulling"]
    dfA = df[np.logical_not(mask)].copy()
    dfB = df[mask].copy()
    dfA[["tree","measurement"]] = dfA["rest"].str.split("_", expand=True)
    dfB[["tree","measurement","pulling"]] = dfB["rest"].str.split("_", expand=True)
    dfA = dfA.loc[:,["date","tree","measurement"]].sort_values(by = ["date","tree","measurement"]).reset_index(drop=True)
    dfB = dfB.loc[:,["date","tree","measurement"]].sort_values(by = ["date","tree","measurement"]).reset_index(drop=True)
    if not dfA.equals(dfB):
        print("Některé měření nemá synchronizaci optika versus tahovky.")
    dfA["type"] = "normal"  # optika je vzdy normal
    return dfA

def get_all_measurements(method='optics', type='normal', *args, **kwargs):
    """
    Find all measurements. If the type is optics, we look for the files from optics only. 
    
    For other types we look for both optics and pulling files and merge informations.
    
    If type is 'all' we return all date. Otherwise only the selected type 
    is returned.
    """
    if method == 'optics':
        return get_all_measurements_optics(*args, **kwargs)
    
    df_o = get_all_measurements_optics()
    df_o["optics"] = True
    # df_p = get_all_measurements_pulling()
    df_p = pd.concat([get_all_measurements_pulling(), get_all_measurements_acc()]).drop_duplicates()
    df = pd.merge(df_p, df_o,
                on=['date', 'tree', 'measurement', "type"], 
                how='left')
    if type != 'all':
        df = df[df["type"]==type]  # jenom vybrany typ mereni
    # df = df[df["tree"].str.contains("BK")]  # neuvazuji jedli
    df["optics"] = df["optics"] == True
    df["day"] = df["date"]
    return df

def available_measurements(df, day, tree, measurement_type, exclude_M01=False):
    """
    Get available measurements for given day, tree and measurement type.
    Allows to exclude M01.
    """
    select_rows = (df["date"]==day) & (df["tree"]==tree)  & (df["type"]==measurement_type)
    values = df[select_rows]["measurement"].values
    if exclude_M01:
        values = [i for i in values if i != "M01"]
    return list(values)

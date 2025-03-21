#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:10:26 2023

@author: marik
"""
import glob
import pandas as pd

def file2data(filename):
    filename = filename.split("/")
    datum = filename[3].replace("_","-")
    filename[-1] = filename[-1].replace(".parquet","")
    strom, mereni = filename[-1].split("_")
    return datum, strom, mereni
  
  
csv_files = glob.glob("../data/parquet/*/BK??_M??.parquet")
csv_files.sort()
csv_files = [file2data(i) for i in csv_files]
df = pd.DataFrame(csv_files, columns=["date","tree", "measurement"])
df["date_tree"] = df["date"]+"_"+df["tree"]

days = df["date"].drop_duplicates().values
days.sort()
days2trees = {}
for day in days:
    days2trees[day] = df[["date","tree"]].drop_duplicates().query('date==@day')["tree"].values

days_trees = df["date_tree"].drop_duplicates().values
day_tree2measurements = {}
for day_tree in days_trees:
    day_tree2measurements[day_tree] = df[["date_tree","measurement"]].drop_duplicates().query('date_tree==@day_tree')["measurement"].values

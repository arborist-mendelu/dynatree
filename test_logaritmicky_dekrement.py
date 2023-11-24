#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:24:49 2023

@author: marik
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from scipy import interpolate
from scipy import stats
from scipy import optimize
import glob
import os

df = pd.read_csv("csv/oscillation_times_remarks.csv")

def peaky (t_ori, y_ori, 
    start=0, end=np.inf, dt = 0.01,
    shift = 0, ini_time = 0.1):
    idx = (t_ori > start-ini_time).ravel() & (t_ori < end).ravel() 
    t = np.arange(start-ini_time,end,dt)
    f = interpolate.interp1d(t_ori, y_ori)
    inishift = f(start) + shift
    y = f(t) - inishift
    intervaly, _ = find_peaks(-np.abs(y), distance = 100)
    intervaly = [i for i in intervaly if np.abs(y[i])<0.1]
    peaky,_ = find_peaks(np.abs(y), distance = 150)
    peaky = peaky[:-1]
    linregres = stats.linregress(t[peaky], np.log(np.abs(y[peaky])))
    output = lambda:None  # https://stackoverflow.com/questions/19476816/creating-an-empty-object-in-python
    output.linregres = linregres
    output.peaks = t[peaky]
    output.intervals = t[intervaly]
    output.f = f
    output.idx = idx
    return output

  
def find_decrement_for_tree(year=2021,month="03",day = "22",
    tree = "BK16",        
    measurement = "M02",
    probe = ("Pt3","Y0"), fixed_by=None, df=df, end=None, start=None):

    row = df.query(f"tree == '{tree}' & measurement == '{measurement}' & date == '{year}-{month}-{day}'")
    if start is None or pd.isna(start):
        if not pd.isna(row['decrement_start']).any():
            start = row['decrement_start'].values[0] 
        else:
            start = row['start'].values[0]
    if end is None or pd.isna(end):
        if not pd.isna(row['decrement_end']).any():
            end = row['decrement_end'].values[0] 
        else:
            end = row['end'].values[0]
    datafile = f"../01_Mereni_Babice_{day}{month}{year}_optika_zpracovani/csv/{tree}_{measurement}.csv"
    data = pd.read_csv(datafile, header=[0,1], index_col=0, dtype = 'float64')
    t_ori = np.array(data["Time"].values).ravel()
    y_ori = np.array(data[probe].values).ravel()
    if fixed_by is None:
        if not pd.isna(row['decrement_fixed_by']).any():
            fixed_by = row['decrement_fixed_by'].values[0]
    if fixed_by is not None:
        y_fix = np.array(data[(fixed_by,"Y0")].values).ravel()
        y_ori = y_ori - y_fix
        
    shift = 0
    res = optimize.minimize_scalar(
          lambda shift: peaky(t_ori, y_ori, start=start, end=end, shift = shift).linregres.rvalue
          )
    out = peaky(t_ori, y_ori, start=start, end=end, shift = res.x)
    
    fig,ax = plt.subplots(figsize=(8,6))
    t = t_ori[out.idx]
    tt = np.arange(start,end,0.01)
    plt.plot(tt,(out.f(tt)-out.f(start)-res.x),label="interpolated data")
    plt.plot(t_ori[out.idx],(y_ori[out.idx]-out.f(start)-res.x),".",label="original data")
    plt.plot(out.intervals,(out.f(out.intervals)-out.f(start)-res.x),"o",label="equilibria")
    plt.plot(out.peaks,(out.f(out.peaks)-out.f(start)-res.x),"o",label="peaks")
    plt.plot(t,np.exp(out.linregres.slope * t)*np.exp(out.linregres.intercept),"--", label="envelope")
    plt.plot(t,-np.exp(out.linregres.slope * t)*np.exp(out.linregres.intercept),"--", label="envelope")
    ax.set(
        # yscale="log",
    #    ylim=(0.1,None)
    )
    ax.legend(loc='upper right')
    fixed_by_text = ""
    if fixed_by is not None:
        fixed_by_text = f"fixed by {fixed_by}"
    ax.set(title=f"{day}.{month}.{year} {tree} {measurement} {probe} {fixed_by_text}")
    plt.grid()
    T = (2*np.diff(out.intervals)).mean()
    freq = (1/(2*np.diff(out.intervals))).mean()
    sd = (1/(2*np.diff(out.intervals))).std()
    decrement = -out.linregres.slope*T
    text = f"Freq = {freq:.5f} ± {sd:.5f}, log. decr = {decrement:.4f}"
    if month in ["06","08"]:
        weight = 800
    else:   
        weight = 300
    t = ax.text(0,0,text,wrap=True, 
            horizontalalignment='left',
            verticalalignment='bottom',
            weight=weight,
            transform=ax.transAxes)
    t.set_bbox(dict(facecolor='yellow', alpha=.7, edgecolor='white'))
    return {'fig':fig, 'decrement': decrement}

def main():
    adresar = "decrement_png"
    if not os.path.exists(adresar):
        os.makedirs(adresar)
    for MEASUREMENT_DAY in [
            "01_Mereni_Babice_22032021_optika_zpracovani",
            "01_Mereni_Babice_29062021_optika_zpracovani",
            "01_Mereni_Babice_05042022_optika_zpracovani",
            "01_Mereni_Babice_16082022_optika_zpracovani",
            ]:
        print(MEASUREMENT_DAY)
        year,month,day = MEASUREMENT_DAY[21:25], MEASUREMENT_DAY[19:21], MEASUREMENT_DAY[17:19]
        soubory = glob.glob(f"../{MEASUREMENT_DAY}/csv/*.csv")
        soubory.sort()
        trees = {}
        for soubor in soubory:
            tree, measurement = soubor.split("/")[-1].split(".")[-2].split("_")
            try:
                output = find_decrement_for_tree(year=year, day=day, month=month, tree=tree, measurement=measurement)
                output['fig'].savefig(f"{adresar}/{tree}_{year}-{month}-{day}_{measurement}.png")
                trees = trees | {tree}
            except:
                pass

find_decrement_for_tree(year=2022, month="08", day="16", tree="BK04", measurement="M03", end=120)
find_decrement_for_tree(year=2022, month="08", day="16", tree="BK04", measurement="M03", end=140)

# if __name__ == "__main__":
#     main()

# find_decrement_for_tree(year=2022, month="08", day="16", tree="BK01", measurement="M03")
# find_decrement_for_tree(year=2022, month="08", day="16", tree="BK01", measurement="M04", end=148)
# find_decrement_for_tree(year=2022, month="04", day="05", tree="BK01", measurement="M04", end=100)
# find_decrement_for_tree(year=2022, month="08", day="16", tree="BK01", measurement="M03", end=130)
# find_decrement_for_tree(year=2022, month="04", day="05", tree="BK01", measurement="M02")

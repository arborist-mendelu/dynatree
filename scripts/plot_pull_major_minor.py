#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 17:12:29 2024

Plot inclinometers from  all data using major/minor setting. 

Useful to check the detection of major/minor axis.

@author: DYNATREE project, ERC CZ no. LL1909 "Tree Dynamics: Understanding of Mechanical Response to Loading"
"""

from dynatree.find_measurements import get_all_measurements_pulling
from dynatree.dynatree import DynatreeMeasurement
import matplotlib.pyplot as plt
import matplotlib
from parallelbar import progress_map

try:
    matplotlib.use('TkAgg') # https://stackoverflow.com/questions/39270988/ice-default-io-error-handler-doing-an-exit-pid-errno-32-when-running
except:
    matplotlib.use('Agg')


df_all = get_all_measurements_pulling()

def plot_one(row):
    m = DynatreeMeasurement(row['date'], row['tree'], row['measurement'], measurement_type = row['type'])
    fig, ax = plt.subplots(figsize=(8,5))
    major_minor = m.identify_major_minor
    df = m.data_pulling
    axes = ["Inclino(80)Maj", "Inclino(81)Maj", "Inclino(80)Min", "Inclino(81)Min"]
    for axis in axes:
        df[axis] = df[major_minor[axis]]
    df.plot(ax=ax, y = axes, style='.', ms=3)
    ax.set(title = f"Pulling data, {m.measurement_type} {m.date} {m.tree} {m.measurement}")
    plt.savefig(f"../temp/inclino/{m.measurement_type}_{m.date}_{m.tree}_{m.measurement}.pdf")
    plt.close('all')

res = progress_map(plot_one, [i for _,i in df_all.iterrows()])

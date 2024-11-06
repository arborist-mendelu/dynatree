#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:08:02 2024

@author: marik
"""

import dynatree
import pandas as pd

def read_data_inclinometers(m, release=None, delta_time=0):
    """
    Read data from pulling tests, restart Time from 0 and turn Time to index.
    If release is given, shift the Time and index columns so that the release 
    is at the given time. In this case the original time is in the column Time_inclino
    
    m .. measurementm instance of DynatreeMeasurement
    
    """
    df_pulling_tests = m.data_pulling
    if "Time" not in df_pulling_tests.columns:
        df_pulling_tests["Time"] = df_pulling_tests.index
    df_pulling_tests["Time"] = df_pulling_tests["Time"] - df_pulling_tests["Time"][0]
    df_pulling_tests.set_index("Time", inplace=True)
    # df_pulling_tests.interpolate(inplace=True, axis=1)
    if release is None:
        return df_pulling_tests
    
    if df_pulling_tests["Force(100)"].isna().all():
        release_time_force = release
    else:
        release_time_force = m.release_time_force
    if release == 0:
        release_time_force = 0
        
    # Sync the dataframe from inclino to optics    
    # if delta_time != 0:
    #     print(f"  info: Using time fix {delta_time} when reading data from inclino/force/elasto")
    df_pulling_tests["Time_inclino"] = df_pulling_tests.index
    shift = - release_time_force + release + delta_time
    dynatree.logger.debug(f"Total shift is {shift}, {release_time_force} (Force), {release} (optics), {delta_time} (delta)")
    df_pulling_tests["Time"] = df_pulling_tests["Time_inclino"] + shift
    df_pulling_tests.set_index("Time", inplace=True)
    df_pulling_tests["Time"] = df_pulling_tests.index

    return df_pulling_tests


def add_horizontal_line(df, second_level=False):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)

    if second_level:
        # Projdi všechny řádky a přidej stylování
        for i in range(1, len(df)):
            if df.index[i][1] != df.index[i - 1][1]:  # Pokud se změní druhý sloupec
                styles.iloc[i, :] = 'border-top: 1px solid black'  # Přidej hranici    
    
    # Projdi všechny řádky a přidej stylování
    for i in range(1, len(df)):
        if df.index[i][0] != df.index[i - 1][0]:  # Pokud se změní první sloupec
            styles.iloc[i, :] = 'border-top: 3px solid red'  # Přidej hranici
    return styles

def ostyluj(subdf, second_level=False):
    vmin=subdf.min(skipna=True).min()
    subdf = (subdf.style.format(precision=3)
             .background_gradient(vmin=vmin, axis=None)
             .apply(lambda x:add_horizontal_line(x, second_level=second_level), axis=None)
             .map(lambda x: 'color: lightgray' if pd.isnull(x) else '')
             .map(lambda x: 'background: transparent' if pd.isnull(x) else '')
             )
    return subdf
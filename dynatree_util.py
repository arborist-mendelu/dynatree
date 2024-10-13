#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 11:08:02 2024

@author: marik
"""

import lib_dynatree

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
    lib_dynatree.logger.debug(f"Total shift is {shift}, {release_time_force} (Force), {release} (optics), {delta_time} (delta)")
    df_pulling_tests["Time"] = df_pulling_tests["Time_inclino"] + shift
    df_pulling_tests.set_index("Time", inplace=True)
    df_pulling_tests["Time"] = df_pulling_tests.index

    return df_pulling_tests
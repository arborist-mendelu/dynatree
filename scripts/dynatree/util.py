#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 07:52:51 2023

Analyzuje pomoci FFT bud jedno mereni nebo celý den nebo všechny dny. 
Je možné načíst jako knihovnu nebo spustit jako program. Pokud je spuštěn jako
program, dělá FFT analýzu pro všechna měření ve všech dnech. 

@author: DYNATREE project, ERC CZ no. LL1909 "Tree Dynamics: Understanding of Mechanical Response to Loading"
"""

import pandas as pd
import numpy as np

# measurement_day = MEASUREMENT_DAY
# path = PATH

def interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index, columns=df.columns)
    df_out.index.name = df.index.name

    for colname in df.columns:
        df_ = df[colname].dropna()
        try:
            df_out[colname] = np.interp(new_index, df_.index, df_.values, right=np.nan)
        except:
            df_out[colname] = np.nan
    return df_out

def extend_dataframe_with_zeros(data, tail=2):
    """
    Input: dataframe with time in the index
    Output: copy of the dataframe with zero signal at the initial and the final part
    
    The mean value is subtracted from the data and zero sequences are
    added on the initial and final part of the data.
    The index is supposed to be time in regular steps. The length of the 
    added sequence is in the variable tail.
    """
    data_pad = data.copy()
    data_pad = data_pad.dropna()
    data_pad = data_pad - data_pad.mean()
    dt = data_pad.index[1]-data_pad.index[0]
    time_start = np.flip(np.arange( data_pad.index[0]-dt, data_pad.index[0]-tail, -dt))
    time_end = np.arange(data_pad.index[-1]+dt, data_pad.index[-1]+tail, dt)
    time_extended = np.concatenate([time_start,time_end])
    df_extended = pd.DataFrame(index = time_extended)
    df_extended[data.columns] = 0
    data_pad = pd.concat([data_pad, df_extended])
    data_pad = data_pad.sort_index()
    return data_pad

def extend_series_with_zeros(data, tail=2):
    """
    Wrapper for extending series by zeros converting to dataframe
    and using extend_dataframe_with_zeros function
    """
    df = pd.DataFrame(data)
    df = extend_dataframe_with_zeros(df, tail=tail)
    return df[df.columns[0]]


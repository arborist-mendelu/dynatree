#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:24:26 2024

@author: marik
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st
import glob
import lib_dynatree as ld
import matplotlib.pyplot as plt
from lib_dynatree import get_all_measurements, get_csv, date2dirname, date2color
st.set_page_config(layout="wide")
from FFT_spectrum import df_remarks, load_data_for_FFT, do_fft_for_one_column, create_fft_image, extend_series_with_zeros

df = get_all_measurements()

#%%
cs = st.columns(2)

with cs[0]:
    """
    ## Day, tree, measurement
    """
    columns = st.columns(3)
    
    with columns[0]:
        day = st.radio("Day",list(df['day'].unique()))
    
    df_day = df[df['day']==day]
    with columns[1]:
        tree = st.radio("Tree",list(df_day['tree'].unique()), horizontal=True)
    
    df_measurement = df_day[df_day['tree']==tree]
    with columns[2]:
        measurement = st.radio("Measurement",list(df_measurement['measurement'].unique()), horizontal=True)
    
    preprocessing_function = st.radio("Preprocessing signal function",["None", "Zeros around, 2 sec", "Zeros around, 4 sec"])
    preprocessing = lambda x:x
    if preprocessing_function == "Zeros around, 2 sec":
        preprocessing = lambda x:extend_series_with_zeros(x,tail=2)
    if preprocessing_function == "Zeros around, 4 sec":
        preprocessing = lambda x:extend_series_with_zeros(x,tail=4)
    
    probe = st.radio("Probe",["Pt3","Pt4"] + [f"BL{i}" for i in range(44,68)] + ["Elasto"], horizontal=True)
    
    "Middle BL is 44-51, side BL are 52-59 (compression) and 60-67 (tension)."

    if probe[0] == "P":
        probe = (probe,"Y0")
    elif probe[0] == "E":
        probe = ("Elasto(90)","")
    else:
        probe = (probe,"Pt0AY")
    date = day
    bounds_for_fft = df_remarks.loc[[(date,f"BK{tree}",f"M0{measurement}")],:]
    start = bounds_for_fft[['start']].iat[0,0]
    end = bounds_for_fft[['end']].iat[0,0]
    if end == np.inf:
        end = 1000
    if pd.isna(end):
        end = 1000
    if pd.isna(start):
        start = 0
    new_start = st.number_input("Signal start",value=start)
    new_end = st.number_input("Signal end",value=end)
    start = new_start
    end = new_end

bounds_for_fft
start,end


    # if pd.isna(start):
    #     print("Start not defined")
    #     continue
    # if start < .1:
    #     print("Start is not set")
    #     continue
st.write(f"{date} BK{tree} M0{measurement} from {start} to {end}, ")        
if probe[0][0]=="E":
    file_csv = f"../{date2dirname(date)}/csv_extended/BK{tree}_M0{measurement}.csv"
    data = load_data_for_FFT(
        file=file_csv,
        start=start,end=end, 
        filter_cols=False, 
        probes=["Time", "Elasto(90)"])
else:
    file_csv = f"../{date2dirname(date)}/csv/BK{tree}_M0{measurement}.csv"
    data = load_data_for_FFT(
        file=file_csv,
        start=start,end=end)

# print(", ",round(data.index[-1]-data.index[0],1)," sec.")

output = do_fft_for_one_column(
    data, 
    probe, 
    preprocessing=preprocessing
    )
if output is None:
    print(f"Probe {probe} failed")
fig = create_fft_image(**output, color=date2color[date])
    
with cs[1]:
    st.write(probe, round(output['peak_position'],6), "±",np.round(output['delta_f'],6))
    st.pyplot(fig)

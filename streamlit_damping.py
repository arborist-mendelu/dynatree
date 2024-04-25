#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 19:31:14 2023

spuštění :  streamlit run streamlit_damping.py

@author: marik
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st
import glob
import lib_dynatree as ld

from lib_dynatree import get_all_measurements, get_csv
from lib_damping import find_damping, get_limits

csv_ans_file = "damping/damping_results.csv"
st.set_page_config(layout="wide")

if 'periods' not in st.session_state:
    ":orange[INFO: Loading dataframe with frequencies to cache.]"
    # code from damping_boxplot.py
    fft_files = glob.glob("fft_data*.xlsx")
    if len(fft_files)==0:
        ":red[ERROR. You need the xlsx files with frequencies to evaluate damping.]"
    dfs = {}
    for i in fft_files:
        day = ld.directory2date(ld.date2dirname(i.split("_")[5]))
        data = pd.read_excel(i)
        data["date"] = day
        data[["tree","measurement"]] = data.iloc[:,0].str.split("_",expand=True)
        dfs[i]=data
        
    df_f = pd.concat(dfs,ignore_index=True)
    df_f = df_f[["date","tree","measurement","Freq"]]   
    df_f.index = pd.MultiIndex.from_frame(df_f[["date","tree","measurement"]])
    df_f = df_f["Freq"]
    st.session_state['periods'] = df_f
else:
    df_f = st.session_state['periods'] 

"""
## Day, tree, measurement
"""

df = get_all_measurements()

#%%
cs = st.columns(2)

with cs[0]:
    columns = st.columns(3)
    
    with columns[0]:
        day = st.radio("Day",list(df['day'].unique()))
    
    df_day = df[df['day']==day]
    with columns[1]:
        tree = st.radio("Tree",list(df_day['tree'].unique()), horizontal=True)
    
    df_measurement = df_day[df_day['tree']==tree]
    with columns[2]:
        measurement = st.radio("Measurement",list(df_measurement['measurement'].unique()), horizontal=True)
    
    if [day,tree,measurement] not in st.session_state:
        # ":orange[INFO: Dataframe data loaded from csv file]"
        df_data = get_csv(day, tree, measurement)
        st.session_state[[day,tree,measurement]] = df_data
    else:
        # ":orange[INFO: Dataframe data from cache]"
        df_data = st.session_state[[day,tree,measurement]]    
    max_cached = 10
    f":orange[INFO: The number of cached measurements: {len(st.session_state)} (cache is cleared automatically the if the number exceeds {max_cached})]"
    if len(st.session_state) > max_cached:
        st.session_state.clear()
    
    with columns[2]:
        probe = st.radio("Probe",["Pt3","Pt4","BL44","BL45","BL46","BL47","BL48"])
        # method= st.radio("Method",["hilbert","peaks"])
    
    start, end, remark = get_limits(date=day, tree=tree, measurement=measurement)
    
    """
    ## Limits
    """
    
    columns = st.columns(3)
    if np.inf == end:
        end = np.nanmax(df_data["Time"].values)
    with columns[0]:
        new_start = st.number_input("From",value=start)
    with columns[1]:
        new_end = st.number_input("To",value=end)
    
    start = new_start
    end = new_end
    
    with columns[2]:
        if st.button('Save'):
            df_times = pd.read_csv("csv/oscillation_times_remarks.csv", index_col=[0,1,2])
            df_times.at[(day,f"BK{tree}",f"M0{measurement}"),"decrement_end"] = end
            df_times.at[(day,f"BK{tree}",f"M0{measurement}"),"decrement_start"] = start
            df_times.to_csv("csv/oscillation_times_remarks.csv")
            st.rerun()
    
    """
    The following table is a record from the file `csv/oscillation_times_remarks.csv`
    The values used in damping computation are in the columns `decrement_start`
    and `decrement_end`. You can override these values by setting From and To
    input fields and pressing Save. The Save button modifies the csv file. If 
    you are happy with the changes, commit new version of csv file to github 
    repository.
    """
    
    remark
    
    """
    ## Results
    """
    sol = {}
    T = 1/(df_f.at[(day,f"BK{tree}",f"M0{measurement}")])
    k = {}
    for method in ['hilbert','peaks']:
        sol[method] = find_damping(date=day, tree=tree, measurement=measurement, 
                       df=df_data, probe=probe, start=start, end=end, method=method)
        k[method] = sol[method]['damping'][0]
    f"""
    * Coefficients $k$ and $q$ from $e^{{kt+q}}$: 
        * Hilbert {sol['hilbert']['damping']}
        * peaks {sol['peaks']['damping']}        
    * Period $T$ from FFT (loaded from xlsx files): {T}
    * **Damping $-kT$ Hilbert transform:** {-k['hilbert']*T}
    * **Damping $-kT$ peak values:** {-k['peaks']*T}
    * **Quotient of dampings:** {k['hilbert']/k['peaks']}
    """

with cs[1]:
    sol['hilbert']['figure']
    sol['peaks']['figure']
    """
    * Blue curve - original signal
    * Orange curve - original signal multiplied by -1
    * Gray curve - envelope from decreasing exponential, nonlinear least squares method
    * Gray dashed curve - envelope from decreasing exponential, linear leas squares method for logaritm of the data
    """
    st.pyplot(sol['hilbert']['figure_fulldomain'])

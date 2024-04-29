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
import matplotlib.pyplot as plt
from lib_dynatree import get_all_measurements, get_csv
from lib_damping import find_damping, get_limits
import emd

csv_ans_file = "damping/damping_results.csv"
st.set_page_config(layout="wide")

if 'periods' not in st.session_state:
    ":orange[INFO: Loading dataframe with frequencies to cache.]"
    # code from damping_boxplot.py
    df_f = pd.read_csv("csv/results_fft.csv", index_col=[0,1,2])
    st.session_state['periods'] = df_f
else:
    df_f = st.session_state['periods'] 


df = get_all_measurements()

#%%
cs = st.columns(4)

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
        probe = st.radio("Probe",["auto","Pt3","Pt4","BL44","BL45","BL46","BL47","BL48"])
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
    T = 1/(df_f.at[(day,f"BK{tree}",f"M0{measurement}"),"Freq"])
    damping = {}
    for method in ['hilbert','peaks','wavelet']:
        sol[method] = find_damping(date=day, tree=tree, measurement=measurement, 
                       df=df_data, probe=probe, start=start, end=end, method=method)
        damping[method] = sol[method]['damping'] 

    container = st.container()
    container.write(f"""
    * Coefficients $k$ and $q$ from $e^{{kt+q}}$: 
        * Hilbert {sol['hilbert']['damping']}
        * peaks {sol['peaks']['damping']}        
    * Period $T$ from FFT (loaded from xlsx files): {T}
    """)

with cs[1]:
    """
    ## Damping using hilbert transform (top) or peaks (bottom)
    """
    sol['hilbert']['figure']
    sol['peaks']['figure']
    """
    * Blue curve - original signal
    * Orange curve - original signal multiplied by -1
    * Gray curve - envelope from decreasing exponential, nonlinear least squares method
    * Gray dashed curve - envelope from decreasing exponential, linear leas squares method for logaritm of the data
    """

with cs[2]:
    """
    ## Wavelet transform
    
    Wavelet transform using the morlet wavelet with frequency corresponding to the 
    first mode of the tree. The initial and final part should be ignored, see the 
    cone of influence, <https://www.mathworks.com/help/wavelet/ug/boundary-effects-and-the-cone-of-influence.html>

    The first picture presents the wavelet transformation. The part before the maximum is
    not considered and equally long part at the and is not considererd as well. (Cone of influence is symmetric.)
    """
    #  wavelet
    st.pyplot(sol['wavelet']['figure'])
    ans = pd.DataFrame(damping,  index=["damping"])
    container.write("* The damping coefficients:")
    container.write(ans)
    container.write("* The relative damping coefficients with respect to Hilbert transformation method:")
    container.write(ans/ans.iloc[0,0])
    
    "The graph of the data. The red part is the part being analyzed."    
    st.pyplot(sol['hilbert']['figure_fulldomain'])  
    
with cs[3]:
    """
    ## Hilbert-Huang transform
    
    See <https://www.mathworks.com/help/signal/ref/hht.html> or <https://emd.readthedocs.io>
    """
    
    dt = 0.01
    time = sol['hilbert']['time']
    data= sol['hilbert']['signal']
    imf = emd.sift.sift(data)
    IP, IF, IA = emd.spectra.frequency_transform(imf, dt, 'hilbert')
    # Define frequency range (low_freq, high_freq, nsteps, spacing)
    freq_range = (0.1, 10, 80, 'log')
    f, hht = emd.spectra.hilberthuang(IF, IA, freq_range, sum_time=False)
    
    for b in [True, False]:
        f"* `sharey` parameter is {str(b)}"
        fig, ax = plt.subplots(figsize=(12,6))
        emd.plotting.plot_imfs(imf, time_vect=time, sharey=b, ax=ax)
        st.pyplot(fig)
    
    # st.pyplot(emd.plotting.plot_imfs(imf, time_vect=time, sharey=False))
    
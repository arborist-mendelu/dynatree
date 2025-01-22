#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 18:51:20 2024

@author: marik
"""

import pandas as pd
import glob
import scipy.io
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import FFT_spectrum as fftdt
import lib_streamlit

fs = 100 # resampled signal
st.set_page_config(layout="wide")

def najdi_soubory_acc_csv(adresar):

    def split_name(i):
        rok,mesic,den,strom,mereni = i.split("-")
        return {'date': f"{rok}-{mesic}-{den}", "tree": strom, "measurement": mereni}

    vzor = os.path.join(adresar, '*.csv')
    seznam_souboru = glob.glob(vzor)
    seznam_souboru = [i.replace(adresar+"/","").replace(".csv","") for i in seznam_souboru]
    return pd.DataFrame([split_name(i) for i in seznam_souboru])

df_files = najdi_soubory_acc_csv("../acc/csv")

c = st.columns(3)

with c[0]:
    # columns = st.columns(3)
    date, tree, measurement = lib_streamlit.get_measurement()
#    tree = f"BK{tree}"
#    measurement = f"M0{measurement}"
    # with columns[2]:
    axis = st.radio("Axis",["X","Y","Z"], horizontal=True)
    subplots= st.radio("Subplots",[True, False, "False and rescale"], horizontal=True)
    tail = st.radio("Tail length",[0,2,4,8,10], horizontal=True)

df = pd.read_csv(f"../data/acc/csv/{date}-{tree}-{measurement}.csv", index_col=0)
df.index = np.arange(df.shape[0])/100
df = df - df.iloc[0,:]
cols = [i for i in df.columns if f"_{axis}" in i.upper()]
df = df[cols]
df_times = pd.read_csv("csv/oscillation_times_acc.csv", index_col=[0,1,2])

time_length = 1.0*df.shape[0]/fs
try:
    start = df_times.at[(date,tree, measurement),"start"] 
    end = df_times.at[(date,tree, measurement),"end"]
    remark = df_times.at[(date,tree, measurement),"remark"]
except:
    start = 0.0
    end = time_length
    remark = ""
with c[0]:
    st.write("Maximum",df.abs().idxmax())
    new_start = st.number_input("Signal start",value=start, step=0.5)
    new_end = st.number_input("Signal end",value=end, step=0.5)
    start = new_start
    end = new_end
    if st.button('Save time'):
        df_times.at[(date,tree, measurement),"start"] = start
        df_times.at[(date,tree, measurement),"end"] = end
        df_times.to_csv("csv/oscillation_times_acc.csv")
        st.rerun()

    remark = st.text_area("Remark",value=remark)
    if st.button('Save time and remark'):
        df_times.at[(date,tree, measurement),"remark"] = remark
        df_times.at[(date,tree, measurement),"start"] = start
        df_times.at[(date,tree, measurement),"end"] = end
        df_times.to_csv("csv/oscillation_times_acc.csv")
        st.rerun()
    if st.button('Save remark'):
        df_times.at[(date,tree, measurement),"remark"] = remark
        df_times.to_csv("csv/oscillation_times_acc.csv")
        st.rerun()

try:
    with c[0]:
        st.write(df_times.loc[[(date,tree, measurement)],:])
        st.write(f"**Remark:**  {remark}")
except:
    pass

with c[0]:
    st.write(f"**Signal length:** {time_length}")

sub_df = df.loc[start:end,:]

if subplots == "False and rescale":
    subplots = False
    sub_df = sub_df / sub_df.max()
figsize = (6.4, 4.8)
if subplots:
    figsize = (6.4, 12.8)    

fig, ax = plt.subplots(figsize=figsize, )
sub_df.plot(subplots=subplots, ax=ax, sharex=True)
plt.suptitle(f"{date}-{tree}-{measurement}")
plt.tight_layout()

with c[1]:
    st.pyplot(fig)

# sub_df = sub_df - sub_df.mean()
with c[2]:
    for col in sub_df.columns:
        fft_output = fftdt.do_fft_for_one_column(sub_df, col, preprocessing=lambda x:fftdt.extend_series_with_zeros(x,tail=tail))
        fft_image = fftdt.create_fft_image(**fft_output)
        fft_image.axes[0].set(title = col)
        st.write(fft_output['peak_position'])
        # st.write(np.mean(fft_output['signal_fft']))
        # fig,ax = plt.subplots()
        # ax.plot(fft_output['signal_fft'])
        # st.pyplot(fig)
        st.pyplot(fft_image)

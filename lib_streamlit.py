#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 10:55:03 2024

@author: marik
"""

import streamlit as st
from lib_dynatree import get_all_measurements

df = get_all_measurements()

def get_measurement():
    
    st.write("""
    ## Day, tree, measurement
    """)
    columns = st.columns(3)
    
    with columns[0]:
        day = st.radio("Day",list(df['day'].unique()))
    
    df_day = df[df['day']==day]
    with columns[1]:
        tree = st.radio("Tree",list(df_day['tree'].unique()), horizontal=True)
    
    df_measurement = df_day[df_day['tree']==tree]
    with columns[2]:
        measurement = st.radio("Measurement",list(df_measurement['measurement'].unique()), horizontal=True)
    return day, tree, measurement
    
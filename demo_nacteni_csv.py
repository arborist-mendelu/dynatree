#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 19:31:14 2023

@author: marik
"""

import pandas as pd

import lib_dynatree

data = lib_dynatree.DynatreeMeasurement("2021-03-22", "BK01", "M03")

data.data_pulling.columns
# df_remarks = pd.read_csv("csv/oscillation_times_remarks.csv")
# df = read_data(f"../01_Mereni_Babice_16082022_optika_zpracovani/csv/BK11_M03.csv")

# #%%

# df[("Pt3","Y0")].plot()

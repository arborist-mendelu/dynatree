#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 19:31:14 2023

@author: marik
"""

import pandas as pd

from lib_dynatree import read_data, read_data_inclinometers

df = read_data_inclinometers("../data/pulling_tests/2021_03_22/BK_01_M2.TXT")


# df_remarks = pd.read_csv("csv/oscillation_times_remarks.csv")
# df = read_data(f"../01_Mereni_Babice_16082022_optika_zpracovani/csv/BK11_M03.csv")

# #%%

# df[("Pt3","Y0")].plot()

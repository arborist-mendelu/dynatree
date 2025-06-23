#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Mon Nov  6 19:31:14 2023

@author: marik
"""

from dynatree import dynatree
import logging
dynatree.logger.setLevel(logging.DEBUG)


# %%
data = dynatree.DynatreeMeasurement("2021-03-22", "BK01", "M03")

d1 = data.data_optics_pt34
d2 = data.data_pulling
# df_remarks = pd.read_csv("csv/oscillation_times_remarks.csv")
# df = read_data(f"../01_Mereni_Babice_16082022_optika_zpracovani/csv/BK11_M03.csv")

# %%

data.data_acc5000

# %%

data.data_acc5000.columns


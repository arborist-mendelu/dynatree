#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Sat Dec  2 22:11:07 2023

@author: marik
"""

from lib_dynatree import read_data, find_release_time_optics

# from csv_add_inclino import extend_one_csv
from plot_probes_inclino_force import plot_one_measurement
from parquet_add_inclino import extend_one_file

from lib_dynatree import DynatreeMeasurement
from static_pull import DynatreeStaticMeasurement
import matplotlib.pyplot as plt
import rich
import lib_dynatree

import logging
lib_dynatree.logger.setLevel(logging.DEBUG)

measurement = "M01"
tree = "BK04"
date = "2022-04-05"
# DF = read_data("../01_Mereni_Babice_22032021_optika_zpracovani/csv/BK04_M02.csv")


f = plot_one_measurement(
        date=date,
        tree=tree, 
        measurement=measurement, 
        # xlim=(0,50),
        # df_extra=df_ext,
        # df=DF,
        return_figure=True, 
        major_minor=False
        ) 
plt.grid()
plt.show()


#%%


m = DynatreeStaticMeasurement(day=date, tree=tree, measurement=measurement)
DF = m.data_optics

m._split_df_static_pulling()

#%%
m._split_df_static_pulling(probe="Pt3")


# %%
pulls = m.pullings

release_time_optics = find_release_time_optics(DF)
release_time_optics

# %%
# rich.inspect(m)

# %%
df_ext = extend_one_file(date=m.date, 
         tree=m.tree, 
         measurement=m.measurement, 
         path="../", 
         write_file=False,
         df=DF
         )  

# df_ext["Time"] = df_ext.index

# %%
f = plot_one_measurement(
        date=date,
        tree=tree, 
        measurement=measurement, 
        # xlim=(0,50),
        # df_extra=df_ext,
        # df=DF,
        return_figure=True, 
        major_minor=True
        ) 
plt.grid()
plt.show()

#%%

d_obj = DynatreeStaticMeasurement(
    day=date, tree=tree,
    measurement="M01", 
    optics=True, 
)
dataset = d_obj.pullings


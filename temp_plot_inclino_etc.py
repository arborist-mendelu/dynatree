#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 22:11:07 2023

@author: marik
"""

from lib_dynatree import read_data
from  csv_add_inclino import extend_one_csv
from plot_probes_inclino_force import plot_one_measurement

# %%
measurement = "M03"
tree = "BK08"
date = "2022-04-05"
# DF = read_data("../01_Mereni_Babice_22032021_optika_zpracovani/csv/BK04_M02.csv")
# %%

# df_ext = extend_one_csv(date=date, 
#         tree=tree, 
#         measurement=measurement, 
#         path="../", 
#         write_csv=False,
#         # df=DF
#         )  
# df_ext["Time"] = df_ext.index

# %%
plot_one_measurement(
        date=date,
        tree=tree, 
        measurement=measurement, 
        # xlim=(0,10),
        # df_extra=df_ext,
        # df=DF
        return_figure=True
        ) 


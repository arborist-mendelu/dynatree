#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:44:47 2024

Srovnani rychlosti cteni pandas a polars. Pandas je vyrazne pomalejsi.
Testovano ctenim z lokalniho SSD disku.

read data
csv read in 2.06 by pandas.
csv read in 0.49 by polars.
pandas takes 4.23 more time.
read data selected
csv read in 0.96 by pandas.
csv read in 0.11 by polars.
pandas takes 8.59 more time.

@author: marik
"""

import lib_dynatree as ld
import time
file = "../data/csv/2021_03_22/BK01_M04.csv"

for f1,f2,comment in [
        [ld.read_data, ld.read_data_by_polars, "read data"],
        [ld.read_data_selected, ld.read_data_selected_by_polars, "read data selected"],
        ]:

    print (comment)
    
    start = time.time()
    df_pandas = f1(file)
    pandas_time = time.time()-start
    print(f"csv read in {pandas_time:.2f} by pandas.")
    
    start = time.time()
    df_polars = f2(file)
    polar_time = time.time()-start
    print(f"csv read in {polar_time:.2f} by polars.")
    print(f"pandas takes {pandas_time/polar_time:.2f} more time.")

#%%

df_polars.values  - df_pandas.values

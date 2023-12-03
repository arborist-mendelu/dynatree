#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 09:07:25 2023

@author: marik
"""

import pandas as pd
import numpy as np
from lib_dynatree import read_data_selected, read_data

# %%time
df = read_data("../01_Mereni_Babice_22032021_optika_zpracovani/csv/BK04_M02.csv")
df["Time"].max()

# %%

%%time

df2= read_data_selected("../01_Mereni_Babice_22032021_optika_zpracovani/csv/BK04_M02.csv")

df2["Time"].max()

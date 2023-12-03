#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 09:07:25 2023

@author: marik
"""

import pandas as pd
import numpy as np
from lib_dynatree import read_data_selected

# %%time
# df = pd.read_csv("../01_Mereni_Babice_22032021_optika_zpracovani/csv/BK04_M02.csv", header=[0,1], dtype=np.float64)

# %%

%%time

df = read_data_selected("../01_Mereni_Babice_22032021_optika_zpracovani/csv/BK04_M02.csv")

df[("Pt3","Y0")].plot()

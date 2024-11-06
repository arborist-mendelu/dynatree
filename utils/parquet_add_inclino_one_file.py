#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: marik
"""

import parquet_add_inclino
import pandas as pd
import lib_dynatree

day = "2021-06-29"
tree = "BK09"
measurement = "M02"
m=lib_dynatree.DynatreeMeasurement(day, tree, measurement)

new_data = parquet_add_inclino.extend_one_file(date=day, tree=tree, measurement=measurement, path="../data", write_file=False)   

ori_data = m.data_optics_extra

ax = new_data["Inclino(80)"].plot(style='.')
ori_data["Inclino(80)"].plot(style='.', ax=ax)

# new_data = new_data.loc[50:60,"Force(100)"]
# ori_data = ori_data.loc[50:60,"Force(100)"]

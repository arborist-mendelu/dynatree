#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:15:06 2024

@author: marik
"""

import static_pull
import lib_dynatree
import logging

lib_dynatree.logger.setLevel(logging.DEBUG)

day, tree, measurement = "2021-03-22", "BK01", "M03"
# day, tree, measurement = "2023-07-17", "BK01", "M03"
cut = 0.3
use_optics = True

data_obj = static_pull.DynatreeStaticMeasurement(
    day=day, tree=tree, measurement=measurement,
    restricted=(cut,0.9), optics=use_optics
    ) 

print(data_obj)

if (not data_obj.is_optics_available) and use_optics:
    lib_dynatree.logger.error(f"{data_obj.date} {data_obj.tree} {data_obj.measurement}: Pozadujes zpracovat optiku, ale data nejsou dostupna.")
    

for i in data_obj.pullings:
    print(i.regressions)

# print(data_obj.data_pulling.shape)
# print(data_obj.data_optics.shape)
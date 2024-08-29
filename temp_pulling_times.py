#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 12:19:28 2024

Pro strom a mereni vypisuje, jak dlouho trvala faze natahovani 30-90 procent maximalni sily. 
Umoznuje srovnat casovou delku mereni pro stejny strom napric jednotlivymi dny.

@author: marik
"""

import lib_dynatree
import static_pull
import logging



days = static_pull.get_all_measurements(method='all')["date"].drop_duplicates()


tree, measurement = "BK01", "M03"

for tree in ["BK01", "BK04", "BK10", "BK14"]:
    print(f"Tree {tree}")
    for day in days:
        data_obj = static_pull.DynatreeStaticMeasurement(day=day, tree=tree, measurement=measurement)
        times = data_obj.pullings[0].data.index
        print (data_obj.pullings, f"interval {(times[-1]-times[0]):.2f}s.")
        
        
measurement = "M01"
for tree in ["BK01", "BK04", "BK10", "BK14"]:
    print(f"Tree {tree}")
    for day in days:
        data_obj = static_pull.DynatreeStaticMeasurement(day=day, tree=tree, measurement=measurement)
        print (f"intervaly:", end=" ")        
        for i,dp in enumerate(data_obj.pullings):
            times = dp.data.index
            print (f"{(times[-1]-times[0]):.2f}", end=", ")
        print()

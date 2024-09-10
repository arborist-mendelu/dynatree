#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 07:38:52 2024

@author: marik
"""

# BK 01 ma nejvetsi uhel (22 stupnu)


import lib_dynatree as ld
import static_pull as sp

measurement_type, tree, measurement, day = "normal", "BK01", "M03", "2021-03-22"

# load the measurement
md = sp.DynatreeStaticMeasurement(day=day, tree=tree, measurement_type=measurement_type, measurement=measurement)
# get the pull phase
angle = 20
h = 0.1
pull = md.pullings[0]
pull.data.columns
treeNo = int(tree[-2:])
pull._process_forces(
    height_of_anchorage= sp.DF_PT_NOTES.at[treeNo,'height_of_anchorage'],
    height_of_pt= sp.DF_PT_NOTES.at[treeNo,'height_of_pt'],
    rope_angle= angle+h,
    height_of_elastometer= sp.DF_PT_NOTES.at[treeNo,'height_of_elastometer'],
    suffix = "angle+h"
    )
pull._process_forces(
    height_of_anchorage= sp.DF_PT_NOTES.at[treeNo,'height_of_anchorage'],
    height_of_pt= sp.DF_PT_NOTES.at[treeNo,'height_of_pt'],
    rope_angle= angle-h,
    height_of_elastometer= sp.DF_PT_NOTES.at[treeNo,'height_of_elastometer'],
    suffix = "angle-h"
    )

collist = ["M_Pt_Measure","Pt3", "Pt4"]
                ]
        else:
            pt_reg = []
        reg = DynatreeStaticPulling._get_regressions(self.data,
            [
            ["M_Rope",   "blue", "yellow", "blue_Maj", "blue_Min", "yellow_Maj", "yellow_Min"],
            ["M_Measure","blue", "yellow", "blue_Maj", "blue_Min", "yellow_Maj", "yellow_Min"],
            ["M_Elasto_Rope", "Elasto-strain"],
            ["M_Elasto_Measure", "Elasto-strain"],
            ]+pt_reg
reg = sp.DynatreeStaticPulling._get_regressions(pull.data, collist)

#%%
data_obj._get_static_pulling_data(restricted=(0.3,0.9), optics=False)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %%
"""
Created on Tue Sep 10 07:38:52 2024

@author: DYNATREE project, ERC CZ no. LL1909 "Tree Dynamics: Understanding of Mechanical Response to Loading"
"""

# BK 01 ma nejvetsi uhel (22 stupnu)


import dynatree.dynatree as ld
import dynatree.static_pull as sp
import pandas as pd
pd.set_option('display.float_format', '{:.6e}'.format)

measurement_type, tree, measurement, day = "normal", "BK01", "M03", "2021-03-22"

h=0.1
def get_regressions(angle = 22, h = 0.1, tree=tree):
    
    # load the measurement
    md = sp.DynatreeStaticMeasurement(day=day, tree=tree, measurement_type=measurement_type, measurement=measurement)
    # get the pull phase
    pull = md.pullings[0]
    # process forces, evaluate moments
    pull._process_forces(
        height_of_anchorage= sp.DF_PT_NOTES.at[(tree,day,measurement_type),'height_of_anchorage'],
        height_of_pt= sp.DF_PT_NOTES.at[(tree,day,measurement_type),'height_of_pt'],
        rope_angle= angle+h,
        height_of_elastometer= sp.DF_PT_NOTES.at[(tree,day,measurement_type),'height_of_elastometer'],
        suffix = ""
        )
    # get regressions
    collist = [
            ["M",   "blue", "yellow"],
            ["M_Elasto", "Elasto-strain"],
            # ["M_Pt","Pt3", "Pt4"]
            ]
    reg = sp.DynatreeStaticPulling._get_regressions(pull.data, collist)
    return reg

def get_derivatives(tree=tree, angle=22, h=0.1):
    result = get_regressions(h=0, tree=tree, angle=angle)[["Independent", "Dependent", "Slope"]]
    plus = get_regressions(h=h, tree=tree, angle=angle)[["Independent", "Dependent", "Slope"]]
    minus = get_regressions(h=-h, tree=tree, angle=angle)[["Independent", "Dependent", "Slope"]]

    result = pd.merge(result, plus, how="left", on=["Independent", "Dependent"], suffixes=('', '_plus'))
    result = pd.merge(result, minus, how="left", on=["Independent", "Dependent"], suffixes=('', '_minus'))
    result["Slope_Derivative"] = (result["Slope_plus"]-result["Slope_minus"])/(2*h)
    result["Slope_Derivative_relative"] = result["Slope_Derivative"]/result["Slope"]
    return result


# %%
bk01 = get_derivatives()
bk01

# %% [markdown]
# Centrální diference s krokem 0.1 stupně ukazuje, že změna úhlu lana o jeden stupeň se projeví na směrnici změnou o 0.7%. Platí pro BK01, který se tahal pod nějvětším úhlem.

# %%
bk04 = get_derivatives(tree="BK04", angle=12)
bk04

# %% [markdown]
# BK04 pod menším úhlem. Změna o jeden stupeň se projeví změnou směrnice o 0.3%.

# %%
bk01['tree'] = "BK01"
bk04['tree'] = "BK04"
ans = pd.concat([bk01,bk04])
ans

# %%
ans.to_excel("../outputs/static_pulling_error_propagation.xlsx")

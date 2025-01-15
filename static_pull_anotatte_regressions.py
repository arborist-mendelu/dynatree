#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 08:36:09 2024

Adds info about failed static pullings data.

@author: marik
"""

import pandas as pd
import numpy as np
import config

df = pd.read_csv(config.file["outputs/regressions_static"], index_col=0)
df_bad = pd.read_csv(config.file["static_fail"])

dfs = {}
for color in ["blue","yellow"]:
    for axis in ["Maj","Min"]:
        d1 = df_bad[df_bad["Dependent"]==color].copy()
        d1.loc[:,"Dependent"] = f"{color}{axis}"
        dfs[f"{color}{axis}"] = d1.copy()
dfs = pd.concat(dfs.values())
df_bad = pd.concat([df_bad, dfs])

#%%
df_bad["pullNo"] = df_bad["pullNo"].astype(int)

# Sloučení obou tabulek na základě sloupců Dependent, tree, measurement, day a pullNo
df_combined = pd.merge(df, df_bad[['Dependent', 'tree', 'measurement', 'day', 'pullNo', 'reason', 'type']], 
                       on=['Dependent', 'tree', 'measurement', 'day', 'pullNo','type'], 
                       how='left', 
                       suffixes=('', '_bad'))

# Přidání sloupce 'failed' (True, pokud se záznam nachází i v df_bad)
df_combined['failed'] = df_combined['reason'].notna()

df_combined['reason'] = df_combined['reason'].fillna('')

df_kamera = pd.read_csv("csv/angles_measured.csv")[["day","tree","type","kamera","nokamera"]]
df_kamera.loc[(df_kamera['day'] == '2024-09-02') & (df_kamera['type'] == 'afterro'), 'type'] = 'afterro2'
for column, value in [["kamera", True], ["nokamera", False]]:
    df_kamera["Dependent"] = df_kamera[column] + "Maj"
    df_kamera.loc[~(df_kamera[column].isna()),column] = value
    df_combined = df_combined.merge(df_kamera[['day', 'tree', 'type', column, 'Dependent']],
                                    on=['day', 'tree', 'type', 'Dependent'],
                                    how='left')
mask = df_combined["nokamera"] == False
df_combined.loc[mask,"kamera"] = False
df_combined = df_combined.drop("nokamera", axis=1)

# Výsledná tabulka
# df_combined[df_combined["failed"]]
df_combined.to_csv(config.file["outputs/anotated_regressions_static"])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 08:36:09 2024

Adds info about failed static pullings data.

@author: marik
"""

import pandas as pd

df = pd.read_csv("../outputs/regressions_static.csv", index_col=0)
df_bad = pd.read_csv("csv/static_fail.csv")
df_bad["pullNo"] = df_bad["pullNo"].astype(int)

# Sloučení obou tabulek na základě sloupců Dependent, tree, measurement, day a pullNo
df_combined = pd.merge(df, df_bad[['Dependent', 'tree', 'measurement', 'day', 'pullNo', 'reason', 'type']], 
                       on=['Dependent', 'tree', 'measurement', 'day', 'pullNo','type'], 
                       how='left', 
                       suffixes=('', '_bad'))

# Přidání sloupce 'failed' (True, pokud se záznam nachází i v df_bad)
df_combined['failed'] = df_combined['reason'].notna()

# Přenesení sloupce Poznamka z df_bad (pouze pro řádky, kde se našla shoda)
df_combined['reason'] = df_combined['reason'].fillna('')

# Výsledná tabulka
df_combined[df_combined["failed"]]
df_combined.to_csv("../outputs/anotated_regressions_static.csv")

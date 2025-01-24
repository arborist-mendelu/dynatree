#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:18:41 2024

@author: marik
"""

import pandas as pd
from dynatree.dynatree import datapath

# Načtení Excel souboru
file_path = f'{datapath}/Popis_Babice_VSE.xlsx'  # Změň cestu k souboru
xls = pd.ExcelFile(file_path)


def fix(sheet_name):
    n = sheet_name.split("_")
    return f"{n[0][4:]}-{n[0][2:4]}-{n[0][:2]}_{n[1]}"

ansdata = {}
# Iterace přes všechny listy v souboru
for sheet_name in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet_name)
    dt = {}
    
    # Procházíme všechny řádky a sloupce v listu
    for row_index, row in df.iterrows():
        # Hledáme všechny sloupce, kde je slovo 'Measurement:'
        for col_index, cell_value in enumerate(row):
            if isinstance(cell_value, str) and 'Measurement:' in cell_value:
                
                # Načteme 6 řádků pod tímto řádkem a 3 sloupce doprava od nalezeného slova
                start_row = row_index + 1
                end_row = row_index + 7  # 6 řádků
                start_col = col_index 
                end_col = col_index + 4 # 3 sousední sloupce
                
                data = df.iloc[start_row:end_row, start_col:end_col].dropna(how='all')
                data = data.reset_index(drop=True)
                data.columns=["measurement","experiment","remark1","remark2"]
                
                # Hledáme 'tree no:' směrem nahoru ve stejném sloupci
                for search_row in range(row_index - 1, -1, -1):
                    if isinstance(df.iloc[search_row, col_index], str) and 'tree no:' in df.iloc[search_row, col_index]:
                        tree_no_value = df.iloc[search_row, col_index + 1]  # Hodnota vedle 'tree no:'
                        break
                
                data["tree"] = tree_no_value
                dt[tree_no_value] = data.dropna(subset=["measurement"])
    if len(dt)>0:
        ansdata[fix(sheet_name)] = pd.concat(dt.values())
        ansdata[fix(sheet_name)]["day"] = fix(sheet_name)
        
df = pd.concat(ansdata.values())

df = (df.loc[:,["day","tree","measurement","remark1","remark2"]]
          .sort_values(by=["day","tree"])
          .reset_index(drop=True)
          )

df.to_csv("csv_output/measurement_notes.csv")

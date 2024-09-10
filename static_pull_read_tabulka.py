#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 05:59:28 2024

@author: marik
"""

import pandas as pd

df = pd.read_excel("../data/Popis_Babice_VSE_13082024.xlsx", sheet_name="Hromadná tabulka")


# Funkce pro doplnění nuly a převod na datum ve formátu YYYY-MM-DD
def preformat_datum(datum):
    # Pokud má datum 7 znaků (např. 3082021), přidáme nulu na začátek dne
    datum = str(datum)
    if len(datum) == 7:
        datum = '0' + datum
    # Převedeme na datetime objekt a následně na požadovaný formát
    return pd.to_datetime(datum, format='%d%m%Y').strftime('%Y-%m-%d')

# Aplikujeme funkci na celý sloupec
df['day'] = df['Datum'].apply(preformat_datum)

df['day'].drop_duplicates().sort_values()

#%%

df["strom"].drop_duplicates()

#%%

df["day"].drop_duplicates().sort_values()

#%%

df["Stav"].drop_duplicates()

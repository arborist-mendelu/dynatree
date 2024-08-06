#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:44:53 2024

Skript hledá tsv soubory a zapisuje do csv souboru adresar s tsv soubory
z optiky pro kazdy datum, strom, mereni.
Ve sloupci files je pocet souboru. 


@author: marik
"""

import glob 
import os
import numpy as np
import re
import pandas as pd

source_dir = "/mnt/ERC/ERC"

adresar = source_dir+'/Mereni_Babice'

# Vytvoření globálního vzoru pro vyhledávání .tsv souborů
vzor = os.path.join(adresar, '**', '*.tsv')

# Nalezení všech .tsv souborů rekurzivně včetně podadresářů
tsv_soubory = glob.glob(vzor, recursive=True)
#%%

def data(text):
    text = re.sub(r'^.*?(Mereni_Babice)', r'\1', text)
    out = text.split("/")[1:5] + [text[-3:]]
    return out

adresare_s_duplikacemi = [os.path.dirname(soubor) for soubor in tsv_soubory]
adresare = set(adresare_s_duplikacemi)
upraveny_seznam = np.array([ data(retezec) + [re.sub(r'^.*?(Mereni_Babice)', r'\1', retezec), adresare_s_duplikacemi.count(retezec)]   
                               for retezec in adresare])

df = pd.DataFrame(upraveny_seznam).drop_duplicates()
df.columns=["date","state","kind","tree","measurement","directory","files"]
nove_poradi = ['date', 'tree', 'measurement', 'state', 'kind', 'directory', 'files']
df = df[nove_poradi]

df = df[df['measurement'].str.startswith('M0')]
df = df.sort_values(by = ["date","tree","measurement"])
df.to_csv("csv/tsv_dirs.csv", index=None)

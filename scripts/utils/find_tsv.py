#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:44:53 2024

Skript hledá tsv soubory a zapisuje do csv souboru adresar s tsv soubory
z optiky pro kazdy datum, strom, mereni.
Ve sloupci files je pocet souboru. 


@author: marik
"""

#%%

import glob 
import os
import numpy as np
import re
import pandas as pd

source_dir = "/mnt/ERC/ERC"
source_dir = "/mnt/ERC2"
filename = "csv/tsv_dirs_ERC2.csv"
adresar = source_dir+'/Mereni_Babice'


#%%
# Vytvoření globálního vzoru pro vyhledávání .tsv souborů
vzor = os.path.join(adresar, '**', '*.tsv')

# Nalezení všech .tsv souborů rekurzivně včetně podadresářů
tsv_soubory = glob.glob(vzor, recursive=True)
#%%

def data(text):
    def extract_m(s):
        m = re.search(r'(M\d+)', s)
        return m.group(1) if m else None
    text = re.sub(r'^.*?(Mereni_Babice)', r'\1', text)
    dirs = text.split("/") 
    out = dirs[1:5] + [extract_m(dirs[-1])]
    return out

# text1 = "/mnt/ERC2/Mereni_Babice/2024_04_10/Bezlisty/Normal/BK09/Optika/zpracovana_data/BK_09_M02_100fps"
# text2 = "/mnt/ERC2/Mereni_Babice/2024_04_10/Bezlisty/Normal/BK01/Optika/zpracovana_data/100fps/BK01_M02"
# #%%
# data(text1)
# #%%
# data(text2)
#%%

adresare_s_duplikacemi = [os.path.dirname(soubor) for soubor in tsv_soubory]
adresare = set(adresare_s_duplikacemi)
upraveny_seznam = np.array([ data(retezec) + [re.sub(r'^.*?(Mereni_Babice)', r'\1', retezec), adresare_s_duplikacemi.count(retezec)]   
                               for retezec in adresare])

df = pd.DataFrame(upraveny_seznam).drop_duplicates()
df = df.dropna()


# df.to_csv("/tmp/find_csv_temp.csv")

#%%
# df = pd.read_csv("/tmp/find_csv_temp.csv")
# print(df)


#%%

df.columns=["date","state","kind","tree","measurement","directory","files"]
nove_poradi = ['date', 'tree', 'measurement', 'state', 'kind', 'directory', 'files']
df = df[nove_poradi]

df = df[df['measurement'].str.startswith('M0')]
df = df[df['directory'].str.contains("zpracovana_data")]
df = df.sort_values(by = ["date","tree","measurement"])
df.to_csv(filename, index=None)

# %%

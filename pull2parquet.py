#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 14:03:16 2024

Cte data ze soubory_z_tahovek.txt vytvoreny napriklad prikazem

find /mnt/ERC/ERC/Mereni_Babice -type f -name "*.TXT" -path "*/Pulling_test/*"

Uklada jako parquet soubory do ../data/parquet_pulling

@author: marik
"""

PREFIX = "/mnt/ERC/ERC/Mereni_Babice/"

import pandas as pd
import multi_handlers_logger
import os
logger = multi_handlers_logger.setup_logger(prefix="pull2parquet")

def get_df():
    
    df = pd.read_csv("soubory_z_tahovek.txt",header=None)
    
    df['path'] = df.iloc[:,0].str.replace(PREFIX,"")
    df = df.drop(0,axis=1)
    buk = df['path'].str.contains("Pulling_test/BK") 
    jedle = df['path'].str.contains("Pulling_test/JD")
    
    df = df[buk|jedle]
    
    # Rozdělíme textový sloupec do nových sloupců podle lomítka
    df_split = df['path'].str.split('/', expand=True)
    df_split.columns = ['day', 'state', 'type', 'tree', 'device', 'filename']
    df = pd.concat([df, df_split], axis=1)
    
    df_split = df["filename"].str.split("_", expand=True)
    df_split.columns = ['tree2', 'measurement']
    df = pd.concat([df, df_split], axis=1)
    df["measurement"] = df["measurement"].str.replace(".TXT","")
    
    # Test, ze jmena souboru odpovidaji adresari a ze je vsude Pulling_test
    test = (df["tree"] == df["tree2"]).all() & (df["device"]=="Pulling_test").all()
    test
    
    df['new_filename'] = "../data/parquet_pulling/"+df["day"]+"/"+df["type"].str.lower()+"_"+df["tree"]+"_"+df["measurement"]+".parquet"
    df['new_directory'] = "../data/parquet_pulling/"+df["day"]
    df["old_filename"] = PREFIX+df["path"]
    return df

#%%
# v cyklu udelej adresar je/li potreba, nacti TXT soubor a uloz parquet soubor
def zpracuj_radek(row):
    """
    Funkce očekává, že row má klíče old_filename, new_filename a new_directory.
    Kontroluje dostupnost zdroje a existenci adresáře a převede starý soubor
    nový. 
    
    Hlásí chybu pokud není zdroj, pokud existuje cíl a pokud se nepodařilo 
    vytvořit adresář.
    """
    source = row['old_filename']
    target = row['new_filename']
    target_dir = row['new_directory']
    if os.path.isfile(source):
        logger.debug(f"Soubor {source} existuje")
    else:
        logger.error(f"Soubor {source} neexistuje")
        return None
    if os.path.isdir(target_dir):
        logger.debug(f"Adresář {target_dir} existuje")
    else:
        logger.info(f"Adresář {target_dir} neexistuje, vytvářím ho")
        try:
            os.mkdir(target_dir)
        except:
            logger.error(f"Adresář {target_dir} se nepodařilo vytvořit")
            return None
    if os.path.isfile(target):
        logger.error(f"Soubor {target} existuje, nic nedělám ...")
        return None
    df = read_csvdata_inclinometers(source)
    df.to_parquet(target)
    logger.info(f"Soubor {target} vytvořen.")
    return None   

def read_csvdata_inclinometers(file):
    """
    Read data from pulling tests. Used to save parquet files.
    """
    df_pulling_tests = pd.read_csv(
        file,
        skiprows=55, 
        decimal=",",
        sep=r'\s+',    
        skipinitialspace=True,
        na_values="-"
        )
    df_pulling_tests["Time"] = df_pulling_tests["Time"] - df_pulling_tests["Time"][0]
    df_pulling_tests.set_index("Time", inplace=True)
    df_pulling_tests = df_pulling_tests.drop(['Nr', 'Year', 'Month', 'Day'], axis=1)
    return df_pulling_tests

#%%
df = get_df()        
for i,row in df.iterrows():
    zpracuj_radek(row)
    






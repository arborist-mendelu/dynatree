#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
!!! IMPORTANT !!!
Pred dalsim spustenim skriptu je potreba upravit cesty podle Patrikova navrhu


Reader for tsv files from xsight. 
Reads tsv files, converts into the form of dataframes with MultiIndex and saves as csv files.

@author: marik
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path
import glob
import re
import config 

def read_tsv_files(cesta, prefix="/mnt/ERC/ERC"):
    """

    Parameters
    ----------
    file path

    Returns
    -------
    Funkce čte tsv soubory z xsightu a transformuje do dataframe s MultiIndexem
    
    """

    ### Definice tabulky s víceúrovňovými nadpisy sloupců, Multiindex
    empty = [[],[]]  # dvě úrovně, na začátku prázdné
    my_index = pd.MultiIndex(levels=empty, codes=empty, names=['source','data'])    
    df = pd.DataFrame(index=my_index).T    

    for file in glob.glob(f"{prefix}/{cesta}/*.tsv"):
        print(".", end="")
        header = re.sub(".*[_/]","",file).replace(".tsv","")
        try:
            df_temp = pd.read_csv(file, sep="\t", dtype = 'float64')
        except:
            try:
                df_temp = pd.read_csv(file, sep="\t", dtype = 'float64', decimal=",")
            except:
                return None
        if df.empty:
            df["Time"] = df_temp["Time [s]"]
        mask = df_temp.columns.str.contains(r'\[\]|\[mm\]',regex=True)
        df_temp = df_temp.loc[:,mask] # selects mask
        nove_sloupce = [i.replace(" []","").replace(" [mm]","").replace(".","") for i in df_temp.columns]
        df_temp = df_temp.set_axis(nove_sloupce, axis="columns", copy=False)
        df[[(header,i) for i in nove_sloupce]] = df_temp
        df = df.copy()
    df.index = df["Time"]
    return df    

#%%


#%%

def main():
    answer = input("The file will convert xsight files to parquet files.\nOlder data (if any) will be replaced.\nConfirm y or yes to continue.")
    # answer = "Y"
    if answer.upper() in ["Y", "YES"]:
        pass
    else:
        print("File processing skipped.")
        return None
    df_tsv = pd.read_csv(config.file["tsv_dirs"])
    for index, row in df_tsv.iterrows():
        if "30fps" in row["directory"]:
            print (f"Skipping 30fps {row['directory']}")
            continue
        print(row["date"])
        date = row['date']
        tree = row['tree']
        measurement = row['measurement']
        if os.path.isfile(f"../data/parquet/{date}/{tree}_{measurement}.parquet"):
            print (f"Soubor ../data/parquet/{date}/{tree}_{measurement}.parquet existuje, nic neprepisuju")
        else:
            try:
                data = read_tsv_files(row["directory"])
                if data is not None:
                    print (f"Vytvářím soubor ../data/parquet/{date}/{tree}_{measurement}.parquet, typ {np.unique(data.dtypes.values)}")
                    if not os.path.isdir(f"../data/parquet/{date}"):
                        os.makedirs(f"../data/parquet/{date}")                    
                    if len(data) > 10:
                        data.to_parquet(f"../data/parquet/{date}/{tree}_{measurement}.parquet")
                        print (f"{len(data)} rows")
                    else:
                        print ("empty")
                else:
                    print (f"Missing or failed tree {tree} and measurement {measurement}.")
            except Exception as e:
                print (f"CHYBA pri zpracovani {date},{tree},{measurement}")
                print (str(e))

if __name__ == "__main__":
    main()





        

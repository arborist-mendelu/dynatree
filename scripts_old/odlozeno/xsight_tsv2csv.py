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

def print_(*args):
    #print(args)
    return None

def print__(*args):
    #print(args)
    return None

experiment_days = [
"01_Mereni_Babice_05042022_optika_zpracovani",
"01_Mereni_Babice_22032021_optika_zpracovani",
"01_Mereni_Babice_29062021_optika_zpracovani",
"01_Mereni_Babice_16082022_optika_zpracovani",
]

trees = ["01","04","07","08","09","10","11","12","13","14","16","21","24"]
measurements = ["02", "03", "04", "05", "06"]

def read_tsv_files(tree, measurement,day="01_Mereni_Babice_22032021_optika_zpracovani"):
    """

    Parameters
    ----------
    tree : Číslo stromu, např "04"
    measurement : Číslo měření, např "02"

    Returns
    -------
    Funkce čte tsv soubory z xsightu a transformuje do dataframe s MultiIndexem
    
    f"../{day}/exports_xsight/BK_{tree}/100fps/BK{tree}_100fps/BK{tree}_M{measurement}/"  
    f"../{day}/exports_xsight/BK_{tree}/BK{tree}_100fps/BK_{tree}_M{measurement}/"
    f"../{day}/exports_xsight/BK_{tree}/BK{tree}_100fps/BK{tree}_M{measurement}/"
    f"../{day}/exports_xsight/BK_{tree}/BK_{tree}_100fps/BK_{tree}_M{measurement}/"    
    """
    cesta = f"../{day}/exports_xsight/BK_{tree}/BK{tree}_100fps/BK{tree}_M{measurement}/"    
    if not os.path.isfile(f"{cesta}/BendLineProbe_BL44.tsv"):
        cesta = f"../{day}/exports_xsight/BK_{tree}/100fps/BK{tree}_100fps/BK{tree}_M{measurement}/"    
        if not os.path.isfile(f"{cesta}/BendLineProbe_BL44.tsv"):
            cesta = f"../{day}/exports_xsight/BK_{tree}/BK{tree}_100fps/BK_{tree}_M{measurement}/"    
            if not os.path.isfile(f"{cesta}/BendLineProbe_BL44.tsv"):
                cesta = f"../{day}/exports_xsight/BK_{tree}/BK_{tree}_100fps/BK_{tree}_M{measurement}/"    
                if not os.path.isfile(f"{cesta}/BendLineProbe_BL44.tsv"):
                    print__ (f"Soubor BL44 {cesta} není k dispozici, asi není zpracováno")
                    return None

    ### Definice tabulky s víceúrovňovými nadpisy sloupců, Multiindex
    empty = [[],[]]  # dvě úrovně, na začátku prázdné
    my_index = pd.MultiIndex(levels=empty, codes=empty, names=['source','data'])    
    df = pd.DataFrame(index=my_index).T    

    for file in glob.glob(f"{cesta}/*.tsv"):
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
        mask = df_temp.columns.str.contains('\[\]|\[mm\]',regex=True)
        df_temp = df_temp.loc[:,mask] # selects mask
        nove_sloupce = [i.replace(" []","").replace(" [mm]","").replace(".","") for i in df_temp.columns]
        df_temp = df_temp.set_axis(nove_sloupce, axis="columns", copy=False)
        df[[(header,i) for i in nove_sloupce]] = df_temp
        df = df.copy()
    return df    

def main():
    answer = input("The file will convert xsight files to csv.\nOlder data (if any) will be replaced.\nConfirm y or yes to continue.")
    if answer.upper() in ["Y", "YES"]:
        pass
    else:
        print("File processing skipped.")
        return None
    for day in experiment_days:
        print ("===================================================")
        print (day)
        for tree in trees:
            print (f"     Tree {tree}")
            for measurement in  measurements:
                print (measurement, " ", end="")
                if os.path.isfile(f"../{day}/csv/BK{tree}_M{measurement}.csv"):
                    print (f"Soubor ../{day}/csv/BK{tree}_M{measurement}.csv existuje, nic neprepisuju")
                else:
                    try:
                        data = read_tsv_files(tree,measurement, day=day)
                        if data is not None:
                            print (f"Vytvářím soubor ../{day}/csv/BK{tree}_M{measurement}.csv, typ {np.unique(data.dtypes.values)}")
                            if not os.path.isdir(f"../{day}/csv"):
                                os.makedirs(f"../{day}/csv")                    
                            if len(data) > 10:
                                data.to_csv(f"../{day}/csv/BK{tree}_M{measurement}.csv")
                                print (f"{len(data)} rows")
                            else:
                                print ("empty")
                        else:
                            print (f"Missing or failed tree {tree} and measurement {measurement}.")
                    except Exception as e:
                        print (f"CHYBA pri zpracovani {day},{tree},{measurement}")
                        print (str(e))

if __name__ == "__main__":
    main()





        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:52:38 2023

POZN: 8.8.2024 byly upraveny cesty pro nove ulozeni souboru. 

Načte inklinoměry, sílu a elastometr, přeškáluje čas na stejné časy
jako v optice, synchronizuje okamžiky vypuštění podle maxima síly a
maxima Pt3.

Čte následující data ve fromatu parquet:

* data z optiky 
* data z tahovek

Zapisuje následující data:

* data z inklinoměrů sesynchronizovaná na optiku do stejneho adresare
  jako data z optiky

Provádí následující činnost:

* Odečte od Pt3,4 pozice Pt11,12,13
* Odečte od Pt1,2 pozice Pt8,9,10
* Snaží se sesynchronizovat obě datové sady.
* Přepočítá sílu, strain a inklinoměry pro stejné časy, jako jsou v
  tabulce s optikou.
* Nové informace zapíše do parquet souboru v adresáři
* Synchronizaci a hodnoty inklinometru je možno doladit pomocí souboru 
  csv/synchronization_finetune_inclinometers_fix.csv 
  Tady je možno opravit synchronizaci a definovat intervaly pro inklinomery, podle
  kterych se ma nastavit nulova hodnota inklinometru.
  
Pokud není naměřená síla, je výstup prázdný (neberou se v úvahu ani 
inklinometry).   --- pozn: toto uz asi neplati

@author: marik
"""

import os
import glob
import pandas as pd
import numpy as np
import warnings
from scipy import interpolate
from lib_dynatree import read_data, find_release_time_optics
from lib_dynatree import read_data_inclinometers, find_finetune_synchro, directory2date
from lib_dynatree import DynatreeMeasurement
from static_pull import DynatreeStaticMeasurement
import lib_dynatree


def fix_data_by_points_on_ground(df):
    """
    Parameters
    ----------
    df : Dataframe to be fixed

    Returns
    -------
    Dataframe with pair Pt0, Pt1 or Pt3, Pt4 fixed by subtracting the coordiantes 
    of Pt8, Pt9, Pt10 (CAM1) and Pt11, Pt12, Pt13 (Cam0)
    Adds columns like Pt0_fixed_by_8. 
    All numbers remain intact.
    """

    by_values = list(range(8,14))
    c = []
    for by in by_values:
        if by in [8, 9, 10]:
            targets = [0, 1]
        elif by in [11, 12, 13]:
            targets = [3, 4]
        c = c + [f"Pt{target}_fixed_by_{by}" for target in targets]   
    
    colindex = pd.MultiIndex.from_product([c, ["X0", "Y0"]], names=["Time", None])    
    df_fixed = pd.DataFrame(index=df.index, columns=colindex)
    for by in range(8,14):
        if by in [8, 9, 10]:
            targets = [0, 1]
        elif by in [11, 12, 13]:
            targets = [3, 4]
        else:
            warnings.warn("The value of by should be one of 8,9,10,11,12,13. The dataframe is kept intact.")
            return df
            
        """
        Calculate the fix. 
        """    
        fix = df.xs(key=f"Pt{by}", level=0, axis=1)
        fix = fix - fix.iloc[0,:]    
        for target in targets:
            for i in ['X0','Y0']:
                df_fixed[(f"Pt{target}_fixed_by_{by}",i)]  = df[(f"Pt{target}",i)] - fix[i]
    return df_fixed

### Test funkce fix_data_by_points_on_ground
# df = read_data("../01_Mereni_Babice_22032021_optika_zpracovani/csv/BK04_M03.csv")
# df_fixed = df.pipe(fix_data_by_points_on_ground)
# plt.plot(df[("Pt3","Y0")])
# plt.plot(df_fixed[("Pt3_fixed_by_11","Y0")])
# plt.plot(df_fixed[("Pt3_fixed_by_12","Y0")])
# plt.plot(df_fixed[("Pt3_fixed_by_13","Y0")])
# find_release_time_optics(df,probe="Pt3",coordinate="Y0")

def resample_data_from_inclinometers(df_pulling_tests, df):
    df_pulling_tests = df_pulling_tests.interpolate(method='index')
    cols = [i for i in df_pulling_tests.columns if "Inclino" in i or "Force" in i or "Elasto" in i or "Rope" in i]
    cols = pd.MultiIndex.from_product([cols,[None]], names=["Time", None])
    df_resampled = pd.DataFrame(index = df.index, columns=cols)
    for col in cols:
        f = interpolate.interp1d(df_pulling_tests.index, df_pulling_tests[col[0]], axis=0, fill_value="extrapolate")
        df_resampled[col] = f(df.index)
    return df_resampled

def extend_one_file(
        date="2021_03_22", 
        tree="01", 
        measurement="2", 
        path="../data", 
        write_file=False, 
        df=None):
    """
    Reads csv file in a csv like directory, 
    adds data from inclinometers (reads from pulling_test subdirectory) and saves to 
    csv_extended subdirectory if write_csv is True. If df is given, the reading of csv 
    file is skipped and the given dataframe is used.

    Returns
    -------
    The dataframe with fixes, inclinometers, force , ...

    """
    lib_dynatree.logger.debug(f"Funkce extend_one_file")
    
    # accept both M02 and 2 as a measurement number
    measurement = measurement[-1]
    # accept both BK04 and 04 as a tree number
    tree = tree[-2:]
    # accepts all "2021_03_22" and "2021-03-22" as measurement_day
    date = date.replace("-","_")    
    
    # Read data file
    # načte data z csv souboru
    if df is None:
        df = read_data(f"{path}/parquet/{date}/BK{tree}_M0{measurement}.parquet")   
    # df se sploupci s odectenim pohybu bodu na zemi
    df_fixed = df.copy().pipe(fix_data_by_points_on_ground) 
    # df s daty z inklinoměrů, synchronizuje a interpoluje na stejné časové okamžiky
    if measurement == "1":
        release_time_optics = 0
        m = DynatreeStaticMeasurement(day=date, tree=tree, measurement=measurement)
        delta_time = 0
    else:
        release_time_optics = find_release_time_optics(df)
        delta_time = find_finetune_synchro(directory2date(date), tree,measurement)
    
    if delta_time != 0:
        print(f"\n  info: Fixing data, nonzero delta time {delta_time} found.")
        
    lib_dynatree.logger.debug(f"    delta time {delta_time} release optics time {release_time_optics}")
    
    # načte synchronizovaná data a přesampluje na stejné časy jako v optice
    m = DynatreeMeasurement(date, tree, measurement)
    df_pulling_tests_ = read_data_inclinometers(
        m, 
        release=release_time_optics, 
        delta_time=delta_time
        )
    df_pulling_tests = resample_data_from_inclinometers(df_pulling_tests_, df)

    # If there is a record related to the Inclinometers, shift values such that 
    # the mean value on the interval given between start and end is zero.    
    list_inclino = ["Inclino(80)X","Inclino(80)Y","Inclino(81)X","Inclino(81)Y"]
    for inclino in list_inclino:
        bounds = find_finetune_synchro(date, tree,measurement, inclino) 
        if bounds is None or np.isnan(bounds).any():
            continue
        start,end = bounds
        print(f"\n  info: Fixing inclinometer {inclino}, setting to zero from {start} to {end}.")
        inclino_mean = df_pulling_tests.loc[start:end,inclino].mean()
        df_pulling_tests[inclino] = df_pulling_tests[inclino] - inclino_mean                

    df_fixed_and_inclino = pd.concat([df_fixed,df_pulling_tests], axis=1)
    if write_file:
        df_fixed_and_inclino.to_parquet(f"{path}/parquet/{date.replace('-','_')}/BK{tree}_M0{measurement}_pulling.parquet")
    return df_fixed_and_inclino

      
def main(path="../data"):
    answer = input("The file will create parquet files with data from inclinometers.\nOlder data (if any) will be replaced.\nConfirm y or yes to continue.")
    # answer = "y"
    if answer.upper() in ["Y", "YES"]:
        pass
    else:
        print("File processing skipped.")
        return None
    files = glob.glob(f"{path}/parquet/*/*.parquet")
    files.sort()
    for file in files:
        date = file.split()
        date,filename = file.split("/")[-2:]
        if not "M01" in filename:
            continue
        print(f"{date}/{filename}, ",end="", flush=True)
        tree = filename[2:4]
        measurement = filename[7]
        extend_one_file(
            date=date, 
            path=path, 
            tree=tree, 
            measurement=measurement,
            write_file=True)
    print(f"Konec zpracování")

if __name__ == "__main__":
    main()


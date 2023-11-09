#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:52:38 2023

Načte inklinoměry, sílu a elastometr, přeškáluje čas na stejné časy
jako v optice, synchronizuje okamžiky vypuštění podle maxima síly a
maxima Pt3.

Čte následující data:
* data z optiky v {measurement_day}/csv/
* data z adresáře {measurement_day}/pulling_tests/

Zapisuje následující data:
* data z inklinoměrů sesynchronizovaná na optiku do
  {measurement_day}/csv_extended/*csv

Provádí následující činnost:
* Odečte od Pt3,4 pozice Pt11,12,13
* Odečte od Pt1,2 pozice Pt8,9,10
* Snaží se sesynchronizovat obě datové sady.
* Přepočítá sílu, strain a inklinoměry pro stejné časy, jako jsou v
  tabulce s optikou.
* Nové informace zapíše do csv souboru v adresáři
  {measurement_day}/csv_extended/. Z důvodu šetření místem a výkonem
  se tabulky nespojují do jedné.
  
Pokud není naměřená síla, je výstup prázdný (neberou se v úvahu ani 
inklinometry).  

@author: marik
"""

import os
import glob
import pandas as pd
import numpy as np
import warnings
from scipy import interpolate
from lib_dynatree import read_data
from lib_dynatree import find_release_time_optics

# read data for synchronization
df_finetune_synchro = pd.read_csv("csv/synchronization_finetune_inclinometers_fix.csv",header=[0,1])

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

def read_data_inclinometers(file, release=None, delta_time=0):
    """
    Read data from pulling tests, restart Time from 0 and turn Time to index.
    If release is given, shift the Time and index columns so that the release 
    is at the given time. In this case the original time in in the column Time_inclino
    """
    df_pulling_tests = pd.read_csv(
        file,
        skiprows=55, 
        decimal=",",
        delim_whitespace=True,    
        skipinitialspace=True,
        na_values="-"
        )
    df_pulling_tests["Time"] = df_pulling_tests["Time"] - df_pulling_tests["Time"][0]
    df_pulling_tests.set_index("Time", inplace=True)
    # df_pulling_tests.interpolate(inplace=True, axis=1)
    if release is None:
        return df_pulling_tests
    
    release_time_force = df_pulling_tests["Force(100)"].idxmax()
        
    # Sync the dataframe from inclino to optics    
    df_pulling_tests["Time_inclino"] = df_pulling_tests.index
    df_pulling_tests["Time"] = df_pulling_tests["Time_inclino"] - release_time_force + release + delta_time
    df_pulling_tests.set_index("Time", inplace=True)
    df_pulling_tests["Time"] = df_pulling_tests.index
        
    return df_pulling_tests

def resample_data_from_inclinometers(df_pulling_tests, df):
    cols = [i for i in df_pulling_tests.columns if "Inclino" in i or "Force" in i or "Elasto" in i or "Rope" in i]
    cols = pd.MultiIndex.from_product([cols,[None]], names=["Time", None])
    df_resampled = pd.DataFrame(index = df.index, columns=cols)
    for col in cols:
        f = interpolate.interp1d(df_pulling_tests.index, df_pulling_tests[col[0]], axis=0, fill_value="extrapolate")
        df_resampled[col] = f(df.index)
    return df_resampled

def extend_one_csv(
        measurement_day="01_Mereni_Babice_22032021_optika_zpracovani", 
        tree="01", 
        tree_measurement="2", 
        path="../", 
        write_csv=False):
    """
    Reads csv file in a csv directory, adds data from inclinometers 
    and saves to csv_extended directory
    

    Returns
    -------
    None.

    """
    # Read data file
    # načte data z csv souboru
    df = read_data(f"{path}{measurement_day}/csv/BK{tree}_M0{tree_measurement}.csv")   
    # df se sploupci s odectenim pohybu bodu na zemi
    df_fixed = df.copy().pipe(fix_data_by_points_on_ground) 
    # df s daty z inklinoměrů, synchronizuje a interpoluje na stejné časové okamžiky
    release_time_optics = find_release_time_optics(df)
    
    # najde případnou opravu synchronizace z tabulky s rucne zadanyma hodnotama
    df_finetune_synchro = pd.read_csv("csv/synchronization_finetune_inclinometers_fix.csv",header=[0,1])    
    condition = (df_finetune_synchro[("tree","-")]==f"BK{tree}") & (df_finetune_synchro[("measurement","-")]==f"M0{tree_measurement}") & (df_finetune_synchro[("date","-")]==f"{measurement_day[21:25]}-{measurement_day[19:21]}-{measurement_day[17:19]}")
    df_finetune_tree = df_finetune_synchro[condition]
    delta_time = 0
    if df_finetune_tree.shape[0]>0:
        delta_time = df_finetune_tree["delta time"].iat[0,0]
    if np.isnan(delta_time):
        delta_time = 0
    
    # načte synchronizvaná data a přesampluje na stejné časy jako v optice
    df_pulling_tests_ = read_data_inclinometers(
        f"{path}{measurement_day}/pulling_tests/BK_{tree}_M{tree_measurement}.TXT", 
        release=release_time_optics, 
        delta_time=delta_time
        )
    df_pulling_tests = resample_data_from_inclinometers(df_pulling_tests_, df)

    # If there is a record related to the Inclinometers, shift values such that 
    # the mean value on the interval given between start and end is zero.    
    list_inclino = ["Inclino(80)X","Inclino(80)Y","Inclino(81)X","Inclino(81)Y"]
    if df_finetune_tree.shape[0]>0:
        for inclino in list_inclino:
            df_finetune_inclino = df_finetune_tree[inclino]
            if not df_finetune_inclino.isnull().values.any():
                inclino_start = df_finetune_inclino["start"].values[0]
                inclino_end = df_finetune_inclino["end"].values[0]
                inclino_mean = df_pulling_tests.loc[inclino_start:inclino_end,inclino].mean()
                df_pulling_tests[inclino] = df_pulling_tests[inclino] - inclino_mean

    df_fixed_and_inclino = pd.concat([df_fixed,df_pulling_tests], axis=1)
    if write_csv:
        df_fixed_and_inclino.to_csv(f"{path}{measurement_day}/csv_extended/BK{tree}_M0{tree_measurement}.csv")
    else:
        return df_fixed_and_inclino

def extend_one_day(measurement_day="01_Mereni_Babice_22032021_optika_zpracovani", path="../"):
    try:
        os.makedirs(f"{path}{measurement_day}/csv_extended")
    except FileExistsError:
       # directory already exists
       pass    
    csvfiles =  glob.glob(f"../{measurement_day}/csv/*.csv")
    csvfiles.sort()
    for file in csvfiles:
        filename = file.split("/")[-1]
        print(filename,", ",end="")
        tree = filename[2:4]
        tree_measurement = filename[7]
        extend_one_csv(
            measurement_day=measurement_day, 
            path=path, 
            tree=tree, 
            tree_measurement=tree_measurement,
            write_csv=True)
    print(f"Konec zpracování pro {measurement_day}")

def main():
    for i in [
                    "01_Mereni_Babice_22032021_optika_zpracovani",   
                    "01_Mereni_Babice_29062021_optika_zpracovani", 
                    "01_Mereni_Babice_05042022_optika_zpracovani",
                    "01_Mereni_Babice_16082022_optika_zpracovani",
                    ]:
        print(i)
        print("=====================================================")
        extend_one_day(measurement_day=i)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skript pomaha rucne nalezt zacatek a konec casoveho intervalu pro FFT.

Skript čte csv/oscillation_times_remarks.csv a zapisuje 
csv/oscillation_times_remarks_new.csv

Skript postupně čte csv soubory a hledá v záznamech csv/oscillation_times_remarks.csv 
nebo v poli new_data informace o začátku a konci intervalu pro FFT. Pokud 
takovou informaci najde, pokračuje s dalším souborem. Pokud informace chybí, 
vykreslí (Pt3,Y0). Je dobré otevřít v okně aby bylo možno zvětšít a odměřovat 
pozice. Takto člověk najde začátek a konec intervalu, který nás zajímá. Tato 
informace se zapíše do proměnné new_data. Poté spust skript znovu a pracuj
s dalším souborem. Takto probereš všechny dosud nezpracované soubory. Pokud 
měření nechceš zpracovávat, dej začátek i konec nuly. Po skončení práce
přejmenuj csv/oscillation_times_remarks_new.csv na 
csv/oscillation_times_remarks.csv a smaž ruční nastavení proměnné new_data.

Na konci zapoznamkuj new_data a prejmenuj csv/oscillation_times_remarks_new.csv 
na csv/oscillation_times_remarks.csv

@author: marik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

csvfiles = glob.glob("../01_Mereni_Babice*optika_zpracovani/csv/*.csv")
csvfiles.sort()

data = pd.read_csv("csv/oscillation_times_remarks.csv", header=0)
data['id'] = data['date'] + " " + data['tree'] + "_" + data["measurement"]

new_data = {}

# new_data['2022-08-16 BK01_M02'] = [start,end,probe if not Pt3,remark]

for datafile in csvfiles:
    datum = datafile.split("_")[3]
    datum = f"{datum[-4:]}-{datum[2:4]}-{datum[:2]}"
    mereni = datafile.split("/")[-1].replace(".csv", "")
    nadpis = f"{datum} {mereni}"
    #print (nadpis)
    if data['id'].str.contains(nadpis).any() or nadpis in new_data.keys():
        continue
    df = pd.read_csv(datafile, header=[0, 1], index_col=0, dtype='float64')
    t = df["Time"].values
    y = df[("Pt3", "Y0")].values
    y2 = df[("Pt4", "Y0")].values
    # y = df[("BL44","Y0")].values  # BL middle
    # # y = df[("BL44","X0")].values  # BL middle
    # y = df[("BL52","Y0")].values  # BL compression side
    y3 = df[("BL60", "Y0")].values  # BL tension side
    y = y - y[0]
    y2 = y2 - y2[0]
    y3 = y3 - y3[0]
    max_idx = np.argmax(np.abs(y))
    # https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    zero_crossings = np.where(np.diff(np.sign(y)))[0]
    idx = zero_crossings > max_idx
    zero_crossings = zero_crossings[idx]
    plt.plot(t, y2, ".", label="Pt4")
    plt.plot(t, y, ".", label="Pt3")
    # plt.plot(t,y3,".",label="BL60, Y0", alpha=0.5)
    plt.plot([t[max_idx]], [y[max_idx]], "o")
    plt.plot(t[zero_crossings], y[zero_crossings], "o")
    plt.grid()
    plt.title(nadpis)
    plt.legend()
    print(f"new_data[\"{nadpis}\"]=[,np.inf,None,None]")

    break

# %%

new_df = pd.DataFrame.from_dict(new_data, orient='index', columns=[
                                "start", "end", "probe", "remark"])

new_df.reset_index(inplace=True)

def getdata(x):
    a, b = x.split(" ")
    c, d = b.split("_")
    return [a, c, d]

new_df[['date', 'tree', 'measurement']] = new_df["index"].apply(
    lambda x: pd.Series(getdata(x)))
new_df.drop(columns=["index"], inplace=True)
data.drop(columns='id', inplace=True)

final_df = pd.concat([data, new_df])

final_df.to_csv("csv/oscillation_times_remarks_new.csv", index=False)



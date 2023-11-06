#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Skript pomaha rucne nalezt zacatek a konec casoveho intervalu pro FFT.

Cte csv/oscillation_times_remarks.csv a zapisuje csv/oscillation_times_remarks_new.csv

Na konci zapoznamkuj new_data a prejmenuj csv/oscillation_times_remarks_new.csv 
na csv/oscillation_times_remarks.csv


@author: marik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob


csvfiles =  glob.glob("../01_Mereni_Babice*optika_zpracovani/csv/*.csv")
csvfiles.sort()

data = pd.read_csv("csv/oscillation_times_remarks.csv", header=0)
data['id'] = data['date'] +" " + data['tree'] + "_" + data["measurement"]

new_data={}

# new_data['2022-08-16 BK01_M02'] = [start,end,probe if not Pt3,remark]

# new_data['2022-08-16 BK01_M02'] = [102.67,np.inf,"Pt4","Pt4 má hezčí průběh"]
# new_data["2022-08-16 BK01_M03"]=[114.4,155.3,None,None]
# new_data["2022-08-16 BK01_M04"]=[130.4,180.5,None,None]
# new_data["2022-08-16 BK04_M02"]=[98.5,133.8,None,None]
# new_data["2022-08-16 BK04_M03"]=[100.8,153.2,None,"interpolovatelné mezery, na konci chybí půlperioda."]
# new_data["2022-08-16 BK04_M04"]=[70.2,np.inf,None,"pěkné kmity, pěkný exponenciální pokles amplitudy"]
# new_data["2022-08-16 BK07_M02"]=[135.7,np.inf,None,"dlouhé kmity, pokles amplitudy, interpolovatelné mezery"]
# new_data["2022-08-16 BK07_M03"]=[86.7,131.1,None,"Zvlnněná střední hodnota, místy hodně mezer, asi interpolovatelné"]
# new_data["2022-08-16 BK07_M04"]=[91.3,np.inf,None,"pěkné kmity, zřetelný exp pokles amplitudy"]
# new_data["2022-08-16 BK07_M05"]=[116.8,np.inf,None,"konec s divným ocáskem"]
# new_data["2022-08-16 BK08_M02"]=[106.02,np.inf,None,"konec s divným ocáskem, amplituda nemá monotonní pokles"]
# new_data["2022-08-16 BK08_M03"]=[75.31,117.23,None,"konec s divným ocáskem"]
# new_data["2022-08-16 BK08_M04"]=[75.54,np.inf,None,"zakolísání při napínání, jinak pěkné kmity s exp. poklesem amplitudy"]
# new_data["2022-08-16 BK09_M02"]=[81.12,np.inf,None,"pěkné kmity, exp. pokles amplitudy"]
# new_data["2022-08-16 BK09_M03"]=[92.10,np.inf,None,"zakolísání při napínání, pěkné kmity"]
# new_data["2022-08-16 BK09_M04"]=[73.55,np.inf,None,"Kmity slábnou a zesilují"]
# new_data["2022-08-16 BK10_M02"]=[67.20,np.inf,None,"Divné vypuštění, pozor na synchronizaci s elasto a inclino. Kmity pěkné, ale není monotonní pokles amplitudy."]
# new_data["2022-08-16 BK10_M03"]=[75.64,np.inf,"Pt4","Pěkné kmity, ale Pt3 ujíždí, použít Pt4."]
# new_data["2022-08-16 BK10_M04"]=[76.4,123.29,None,"Velké zakolísání při napínání, střední honodta při kmitání se mění."]
# new_data["2022-08-16 BK11_M02"]=[111.24,np.inf,None,"Pěkné kmity, pěkný pokles."]
# new_data["2022-08-16 BK11_M03"]=[105.45,np.inf,None,"Skok v čase 145"]
# new_data["2022-08-16 BK11_M04"]=[85.87,140.07,None,"Končí zvlněným ocáskem."]
# new_data["2022-08-16 BK12_M02"]=[61.6,np.inf,None,"Pěkné kmity, jedna interpolovatelná mezera."]
# new_data["2022-08-16 BK12_M03"]=[87.63,np.inf,None,"Dvě zakolísání při napínání, v kmitech interpolovatelná mezera."]
# new_data["2022-08-16 BK12_M04"]=[66.44,np.inf,None,"Roztřepené napínání, kmity pěkné."]
# new_data["2022-08-16 BK13_M02"]=[0,0,None,"Nepovedené, nebrat"]
# new_data["2022-08-16 BK13_M03"]=[94.51,np.inf,None,"Pěkné, rychlý útlum."]
# new_data["2022-08-16 BK13_M04"]=[125.1,np.inf,None,"Pěkné, rychlý útlum, na konci ocásek."]
# new_data["2022-08-16 BK16_M02"]=[55.38,np.inf,None,"Roztřepené napínání"]
# new_data["2022-08-16 BK16_M03"]=[71.99,np.inf,None,"Pěkné kmity."]
# new_data["2022-08-16 BK16_M04"]=[65.8,np.inf,None,"Konec s ocáskem."]
# new_data["2022-08-16 BK21_M02"]=[144.10,np.inf,None,"Krásné kmity, rychle utlumené."]
# new_data["2022-08-16 BK24_M02"]=[77.94,np.inf,None,None]
# new_data["2022-08-16 BK24_M03"]=[110.2,np.inf,None,"Krásné kmity, rychle utlumené."]
# new_data["2021-03-22 BK13_M04"]=[86.31,119.98,None,"Interpolovatelné mezery, změna tvaru peaků."]

for datafile in csvfiles:
    datum = datafile.split("_")[3]
    datum = f"{datum[-4:]}-{datum[2:4]}-{datum[:2]}"
    mereni = datafile.split("/")[-1].replace(".csv","")
    nadpis = f"{datum} {mereni}"
    #print (nadpis)
    if data['id'].str.contains(nadpis).any() or nadpis in new_data.keys():
        continue
    df = pd.read_csv(datafile, header=[0,1], index_col=0, dtype = 'float64')
    t = df["Time"].values
    y = df[("Pt3","Y0")].values
    y2 = df[("Pt4","Y0")].values
    # y = df[("BL44","Y0")].values  # BL middle
    # # y = df[("BL44","X0")].values  # BL middle
    # y = df[("BL52","Y0")].values  # BL compression side
    y3 = df[("BL60","Y0")].values  # BL tension side
    y = y-y[0]
    y2 = y2 - y2[0]
    y3 = y3 - y3[0]
    max_idx = np.argmax(np.abs(y))
    # https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    zero_crossings = np.where(np.diff(np.sign(y)))[0]
    idx = zero_crossings > max_idx
    zero_crossings = zero_crossings[idx]
    plt.plot(t,y2,".",label="Pt4")
    plt.plot(t, y,".",label="Pt3")
    # plt.plot(t,y3,".",label="BL60, Y0", alpha=0.5)
    plt.plot([t[max_idx]],[y[max_idx]],"o")
    plt.plot(t[zero_crossings],y[zero_crossings],"o")
    plt.grid()
    plt.title(nadpis)
    plt.legend()
    print (f"new_data[\"{nadpis}\"]=[,np.inf,None,None]")
    
    break

# %%

new_df = pd.DataFrame.from_dict(new_data, orient='index', columns = ["start","end","probe","remark"])

new_df.reset_index(inplace=True)
def getdata(x):
    a,b = x.split(" ")
    c,d = b.split("_")
    return [a,c,d]

new_df[['date', 'tree', 'measurement']] = new_df["index"].apply(lambda x: pd.Series(getdata(x)))
new_df.drop(columns=["index"], inplace=True)
data.drop(columns='id', inplace=True)

final_df = pd.concat([data, new_df])

final_df.to_csv("csv/oscillation_times_remarks_new.csv", index=False)


# new_df['date','tree','measurement'] = new_df.loc[:,'index'].apply(getdata, result_type='expand')

# output_df = pd.DataFrame(data).T
# output_df.columns = ["start","end","remark"]
# output_df.reset_index(inplace=True)
# output_df[['date', 'measurement_']] = output_df["index"].apply(lambda x: pd.Series(str(x).split(" ")))
# output_df[['tree', 'measurement']] = output_df["measurement_"].apply(lambda x: pd.Series(str(x).split("_")))
# output_df.drop(columns=["index","measurement_"], inplace=True)
# output_df = output_df[["date","tree", "measurement", "start", "end", "remark"]]
# output_df["date"] =  pd.to_datetime(output_df["date"], format='%Y-%m-%d')
# output_df = output_df.astype({'tree':'string', 'measurement':'string',})
# output_df.sort_values(by = ["date","tree","measurement"], inplace=True)

# output_df.reset_index(drop=True, inplace=True)
# print(output_df.columns)
# print(output_df.dtypes)
    
# print(output_df.head())



# %% SCHOVANO

# rucne urcene meze pro kmity
# data = {
# "2021-06-29 BK08_M04":[ 65.06,  86.76, "Roztřepaná křivka při natahování, zmatek na konci odříznut"],
# "2021-06-29 BK13_M02":[ 91.74, 123.73, "Oscilace při natahování, relativně pěkné oscilace"],
# "2021-06-29 BK24_M04":[128.50, 165.46, "Chybí dva celé obloučky !!!"],
# "2021-06-29 BK16_M02":[ 58.37,  74.44, "Oscilace při náběhu, chybí půlperioda uprostřed, konec rozsypaný"],
# "2021-06-29 BK21_M04":[ 98.76, 139.70, "Rozsypaný náběh, ale jinak pěkné, uprostřed kousek chybí"],
# "2021-06-29 BK24_M03":[ 94.10, 129.19, "Oscilace pěkné, chybí kus uprostřed"],
# "2021-06-29 BK01_M04":[ 55.04,  80.70, "Hrbatý náběh, oscilace mají zvláštní tvar, kousek chybí"],
# "2021-06-29 BK12_M03":[ None, None, "Úplně nepovedené"], 
# "2021-06-29 BK12_M06":[ 38.00,  73.49, "Divný průběh napínání, oscilace pěkné, kus chybí, ale mohlo by se dorovnat interpolací"],
# "2021-06-29 BK13_M03":[ 86.82, 121.33, "Oscilace při napínání, dva kusy uprostřed chybí"],
# "2021-06-29 BK08_M02":[ 53.07,  68.28, "Roztřepená křivka při napínání, oscilace relativně OK, ale je jich málo"], 
# "2021-06-29 BK12_M04":[ 49.68,  82.72, "Oscilace pěkné, ale kus chybí, snad by se dalo dointerpolovat, zuby při napínání"], 
# "2021-06-29 BK01_M03":[ 49.80,  70.74, "Mezery v datech při oscilaci, jina pěkné ale málo period"],
# "2021-06-29 BK07_M04":[ 84.40,  96.76, "Zubaté napínání, divné kmity, s mezerou"], 
# "2021-06-29 BK10_M04":[ 46.53,  80.90, "Pěkné kmity s mezerou (snad se dá dointerpolovat), zubaté napínání"], 
# "2021-06-29 BK12_M05":[ 61.36, 106.26, "Oscilace na začátku, zuby při napínání, interpolovatelné mezery v datech, "],
# "2021-06-29 BK09_M04":[ 64.81,  96.76, "Hodně roztřepené, i při napínání"],
# "2021-06-29 BK14_M03":[ None, None, "Vůbec nevím, co to je"],
# "2021-06-29 BK01_M02":[ 49.28,  64.48, "Použitelné jenom tři periody, třetí je navíc divná"],
# "2021-06-29 BK14_M05":[ 72.11, 108, "Obsahuje hodně mezer, neinterpolovatelné"],
# "2021-06-29 BK09_M02":[ 36.00, 70.74, "Lehce rotřepené oscilace, ale použitelné"],
# "2021-06-29 BK11_M05":[ 51.62, 89.26, "Na konci dlouhá mezera"],
# "2021-06-29 BK01_M05":[ 57.92, 80.7, "Konec roztřepený"],
# "2021-06-29 BK11_M03":[ None, None, "Co to je?"],
# "2021-06-29 BK21_M02":[ 85.54, 119.86, "Jedna půlperioda chybí, jinak asi OK ale málo kmitů"],
# "2021-06-29 BK13_M05":[ 94.72, 132.43, "Oscilace při natahování, dlouhé neinterpolovatelné mezery v datech"],
# "2021-06-29 BK04_M03":[ 36.83, 68.07, "Málo period, rychle utlumené, ale asi použitelné"],
# "2021-06-29 BK16_M04":[ 48.9,  92.74, "Měření končí poměrně brzo, ještě při velkých výchylkách"],
# "2021-06-29 BK10_M03":[ 49.9, 79.83, "Zubatý náběh, interpolovatelné mezery v datech"],
# "2021-06-29 BK11_M04":[ 73.82, 104.85, "Oscilace OK, ale asi neinterpolovatelné mezery v datech"],
# "2021-06-29 BK14_M06":[ 70.52, 107.89, "Neinterpolovatelné mezery v datech, zřetelné zhoupnutí po průchodu rovnovážnou polohou"],
# "2021-06-29 BK21_M05":[108.6, 152.35, "Kmity pěkné, ale začátek kmitů nepoužitelný, možná použít jiný probe"],
# "2021-06-29 BK07_M03":[70.54, 105.05, "Pěkné kmity, konec horší"],
# "2021-06-29 BK10_M02":[39.30, 69.85, "Zubatý náběh, pěkné kmity s interpolovatelnými mezerami"],
# "2021-06-29 BK09_M03":[60.03, 90.32, "Zubaté, ale použitelné"],
# "2021-06-29 BK07_M05":[65.47, 96.47, "Zubaté už od náběhu, interpolovatelné mezery v datech, konec divný ale pro jiný probe možná bude OK"],
# "2021-06-29 BK04_M04":[34.66, 76.80, "Pěkné oscilace, dostatek period, interpolovatelné mezery, krásné obloučky"],
# "2021-06-29 BK21_M03":[65.50, 98.07, "Pěkné oscilace, pěkné obloučky, dost dat chybí, možná použít vhodný probe"],
# "2021-06-29 BK07_M02":[79.09, 111.38, "Pěkné oscilace, při natahování nějaké rozkolísání, ale jinak OK"],
# "2021-06-29 BK14_M02":[88.52, 112.64, "Zubaté napínání, řídká data"],
# "2021-06-29 BK04_M02":[46.63, 77.12, "Nekmitá okolo rovnovážné polohy"],
# "2021-06-29 BK08_M05":[57.46, 81.13, "Zubaté napínání, řídká data, ale asi použitelné"],
# "2021-06-29 BK12_M02":[44.84, 74.28, "Oscilace při napínání, interpolovatenléné mezery, nekmitá okolo rovnovážné polohy"],
# "2021-06-29 BK16_M03":[34.24, 62.37, "Zubaté napínání, uprostřed hodně chybí, zřetelná vratka po vypuštění"],
# "2021-06-29 BK24_M02":[99.95, 127.24, "Pěkné napínání i kmity, ale některá data chybí, možná použít jiný probe"],
# "2021-06-29 BK13_M04":[75.11, 113.45, "Velmi divoký průběh celého experimentu od začátku do konce, ASI NEPOUŽITELNÉ"],
# "2021-06-29 BK08_M03":[33.20, 58.04, "Zubaté napínání, ale kmity pěkné a hladké"],
# "2021-06-29 BK11_M02":[59.54, 100.26, "Pěkné kmity, dostatek period, interpolovatelné mezery"], 
# "2021-06-29 BK14_M04":[71.17, 99.08, "Zhoupnuti po pruchodu rovnovaznou polohou"],
# "2022-04-05 BK08_M04":[55.38, 85.53, "Pěkné kmity, na konci podivný ocásek"],
# "2022-04-05 BK16_M02":[None, None, "Osciluje už napínání, po vypuštění divné kmity"],
# "2022-04-05 BK12_M03":[None, None, "Osciluje už napínání, divné tvaru peaků, neosciluje okolo rovnovážné polohy, rozsypaný čaj ve druhé periodě"],
# "2022-04-05 BK08_M02":[54.9, 114.39, "Na konci se opět zesiluje, ale napínání i oscilace pěkné"],
# "2022-04-05 BK12_M04":[59.83, 108.08, "Osciluje už náběh, po vypuštění divné tvarz peaků, amplituda neklesá, neosciluje okolo rovnovážného stavu"],
# "2022-04-05 BK07_M04":[77.97, 125.5, "Řídká (ale použitelná) data, po čtyřech periodách divný průběh, konec rozsypaný"],
# "2022-04-05 BK10_M04":[72.24, 118.71, "Oscilace při napínání, pěkné kmity, dostatek period, na konci zvláštní amplituda a tvar peaků"],
# "2022-04-05 BK09_M04":[63.29, 123.79, "Kmitani se neustálí, ke konci amplituda neklesá"],
# "2022-04-05 BK14_M03":[44.32, 93.8, "Velke kmity i v pocatecni fazi i na konci, asi velky vitr"],
# "2022-04-05 BK14_M05":[65.62, 115.18, "Velke kmity na konci, vitr?"],
# "2022-04-05 BK09_M02":[71.28, 113.68, "Pekne kmity i v pocatecni fazi"],
# "2022-04-05 BK11_M03":[56.26, 107.20, "Zubaté napínání, interpolovatelné mezery, ke konci amplituda neklesá, ale vypadá použitelně"],
# "2022-04-05 BK21_M02":[98.26, 158.25, "Krasné dlouhé kmity"],
# "2022-04-05 BK04_M03":[71.70, 131.24, "Pravidelné kmity i na začátku, neustálí se, asi vítr"], 
# "2022-04-05 BK16_M04":[50.39, 100.78, "Velké kmity, neklesá amplituda, zřetelné kmity i při napínání, asi vítr"],
# "2022-04-05 BK10_M03":[57.28, 104.40, "Velmi nepravidelný průběh"],
# "2022-04-05 BK11_M04":[60.21, 93.44, "Amplituda klesá pomalu, napravidelnosti při natahování, asi vítr Máchale"],
# "2022-04-05 BK07_M03":[59.03, 97.79, "Kmity během natahování i na konci"],
# "2022-04-05 BK10_M02":[53.05, 102.16, "Po ustálení se znovu rozkmitá"],
# "2022-04-05 BK09_M03":[52.20, 110.56, "Kmity pěkné, ale moc neslábnou a je hodně nahusto ale s malou frekvencí rozkmitaný počátek"],
# "2022-04-05 BK04_M04":[70.12, 131.98, "Kmity pěkné"],   
# "2022-04-05 BK07_M02":[157.33, 217.70, "Kmity pěkné, ale mají mezeru cca 2 vteřiny"],
# "2022-04-05 BK14_M02":[79.0, 127.65, "Kmitá i během natahování, hodně kmitá i nepravidelně, neustálí se"],
# "2022-04-05 BK04_M02":[98.89, 152.30, "Kmitá i během natahování, po vypustění se ustálí a zesílí"],
# "2022-04-05 BK12_M02":[41.54, 91.5, "Druhá polovina je poměrně chaotická, asi vítr"],
# "2022-04-05 BK16_M03":[56.74, 106.24, "Kmity během natahování, nepravidelnosti v amplitudě, asi vítr"],
# "2022-04-05 BK24_M02":[58.24, 109.44, "Krásné kmity"],
# "2022-04-05 BK08_M03":[58.0, 104.95, "Drobné kmity během napínání, chaos na konci"],
# "2022-04-05 BK11_M02":[68.3, 119.44, "Hodně pěkných kmitů, na konci ale chaos"],
# "2022-04-05 BK14_M04":[49.86, 98.2, "Kmity během napínání, nepravidelnosti na konci"],
# "2021-03-22 BK08_M04":[60.05, 100.34, "Pěkné kmity, jenom trochu chlupaté"],
# "2021-03-22 BK13_M02":[59.43, np.inf, "Pěkné kmity, divný tvar maxim a minim"],
# "2021-03-22 BK24_M04":[91.82, 143.28, "Pěkné kmity"],
# "2021-03-22 BK16_M02":[32.18, 63.16, "Pěkné kmity, jenom divný tvar prvního peaku"],
# "2021-03-22 BK21_M04":[72.30, 124.8, "Velice pěkné kmity, dlouhý vzorek"],
# "2021-03-22 BK24_M03":[93.97, np.inf, "Pěkné kmity"],
# "2021-03-22 BK01_M04":[44.34, 97.42, "Pěkné kmity"],
# "2021-03-22 BK12_M03":[None, None, "Dost nepovedené, divný tvar kmitů a jenom několik zakmitnutí"],
# "2021-03-22 BK13_M03":[74.34, 117.71, "Divné vypuštění, zvláštní tvar peaků, na konci roste amplituda"],
# "2021-03-22 BK08_M02":[29.24, 61.09, "Na konci roste amplituda"],
# "2021-03-22 BK12_M04":[23.9, 50.20, "Divný tvar kmitů, divné vypuštění, krátká fáze natahování"],
# "2021-03-22 BK01_M03":[50.83, 105.59, "Krásné kmity, dlouhý vzorek"],
# "2021-03-22 BK07_M04":[42.25, 77.6, "Hrbaté natahování, odříznutý divný konec, kmity moc neslábnou"],
# "2021-03-22 BK10_M04":[46.18, 69.3, "Zašumněné natahování, lehce divný tvar prvních peaků"],
# "2021-03-22 BK09_M04":[41.85, 71.47, "Zašumněné natahování, divný tvar prvních peaků"],
# "2021-03-22 BK14_M03":[53.24, 80.12, "Zašumněné natahování i vypuštění"],
# "2021-03-22 BK01_M02":[63.4, 70.54, "Jenom dvě periody, ale pěkné zakolísání po vypuštění"],
# "2021-03-22 BK09_M02":[33.07, 56.45, "Zašumněné natahování i konec, zvláštní tvar peaků"],
# "2021-03-22 BK11_M03":[49.97, 76.98, "Patrné oscilace při natahování"],
# "2021-03-22 BK08_M06":[33.01, np.inf, "Patrné oscilace při natahování"],
# "2021-03-22 BK21_M02":[71.81, np.inf, "Pěkné oscilace"],
# "2021-03-22 BK04_M03":[40.63, 90.90, "Pěkné oscilace"],
# "2021-03-22 BK16_M04":[50.61, 80.33, "Divné chování na začátku a hned po vypuštění odfiltrováno"],
# "2021-03-22 BK10_M03":[43.88, 70.07, "Roztřesené natahování, velmi zvláštní tvar peaků"],
# "2021-03-22 BK11_M04":[57.07, 93.16, "Pěkné oscilace"],
# "2021-03-22 BK07_M03":[62.66, 113.70, "Pěkné oscilace"],
# "2021-03-22 BK10_M02":[45.33, np.inf, "Pěkné oscilace, kousek přerušení, zvláštní tvar peaků"],
# "2021-03-22 BK09_M03":[39.10, 68.92, "Oscilace OK, divný tvar peaků, konec odřezaný kvůli roztřesení, peaky během natahování a před vypuštěním"],
# "2021-03-22 BK07_M05":[66.58, 117.80, "Pěkné oscilace na dlouhém intervalu"],
# "2021-03-22 BK04_M04":[40.36, 97.21, "Kmity neslábnou, jeden peak rozsypaný, Pt3 končí moc brzo (použít jiný probe)"],
# "2021-03-22 BK21_M03":[64.89, 113.63, "Pěkný experiment, hodně period, hladká křivka, mezery vypadají interpolovatelně"],
# "2021-03-22 BK07_M02":[119.91, 183.49, "Hodně kmitů, zřetelný exponenciální pokles amplitudy, řídká snad interpolovatelná data"],
# "2021-03-22 BK14_M02":[39.17, np.inf, "Chlupaté ale použitelné, zřetelný exp. pokled amplitudy"],
# "2021-03-22 BK04_M02":[51.13, 99.37, "Oscilace na začátku napínání, jinak pěkné oscilace a dostatek period, zřetelný exp. pokles amplitudy"],
# "2021-03-22 BK08_M05":[47.44, np.inf, "Chlupato-zubaté napínání, kmity OK s ocasem na konci, zřetelný exp. pokles amplitudy"],
# "2021-03-22 BK12_M02":[50.56, np.inf, "Experiment OK, ale neobvyklé tvary peaků"],
# "2021-03-22 BK16_M03":[47.73, 71.61, "Chlupaté napínání ale OK, divný tvar prvního peaku"],
# "2021-03-22 BK24_M02":[117.54, np.inf, "Pěkná křivka, interpolovatelné mezery, zřetelný exp. pokles amplitudy"],
# "2021-03-22 BK08_M03":[63.22, np.inf, "Relativně pěkná křivka"],
# "2021-03-22 BK11_M02":[41.82, np.inf, "Hodně period, ale řídká interpolovatelná data, amplituda klesá spíš lineárně"],
# "2021-03-22 BK14_M04":[54.84, 80.21, "Zježený náběh, ale použitelné"],
# "2022-04-05 BK13_M02":[118.3, 160.95, "Pravidelné oscilace při napínání, na konci zvýšení amplitudy - část ponechána a část odřezána"],
# "2022-04-05 BK24_M04":[51.0, 111.07, "Při natahování oscilace, na konci kolísá střední hodnota, hodně kmitů; kousek chybí, ale intrerpolovatelné"],
# "2022-04-05 BK21_M04":[93.30, 142.84, "Hladké napínání, pěkné kmity, zřetelný útlum, dostatek period, interplovatelné mezírky"],
# "2022-04-05 BK24_M03":[82.38, 139.46, "Před vypuštěním dlouhá pauza, rozbitý naviják, interpolovatelné mezery, střední hodnota se lehce vlní"],
# "2022-04-05 BK01_M04":[72.7, 137.09, "Vypuštění je divočejší, první výchylka neobvykle velká, jinak oscilace pěkné, mezery interpolovatelné"],
# "2022-04-05 BK13_M03":[86.74, 135.81, "Oscilace při natahování, ke konci kmity neslábnou"],
# "2022-04-05 BK01_M03":[71.51, 110.15, "Oscilace i při natahování, střední hodnota kolísá více než je amplituda kmitů, ale Pt7 je poměrně stabilní, kamera asi nepadala"],
# "2022-04-05 BK01_M02":[93.64, 138.15,"Mírné a pěkné oscilace při začátku natahování, hezké oscilace po vypuštění, na konci tlustou čarou (rozkolísané) ale pořád pěkné" ],
# "2022-04-05 BK21_M03":[102.3, 147.2, "Pt3 se ztratí hned při vypuštění, BL44 po vypuštění malý kmit, potom velký a tlumení k nenulové hodnotě, Pt7 u kmene se při vypuštění pohnul, oscilace při napínání, mezery s mizernou vyhlídkou na interpolaci"],
# "2022-04-05 BK13_M04":[67.57, 115.76, "Pěkné oscilace při napínání, slábnoucí a sílící oscilace při kmitání, případné mezery asi interpolovatelné"],
# }

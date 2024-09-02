#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:24:26 2024

@author: marik
"""

import plotly.express as px
import solara
import solara_select_source as s
import lib_dynatree
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq

DT = 0.01

def set_click_data(x=None):
    if x['device_state']['shift']:
        t_to.value = x['points']['xs'][0]
    else:
        t_from.value = x['points']['xs'][0]

def plot():
    # click_data, set_click_data = solara.use_state(None)
    # click_data = None
    data_obj = lib_dynatree.DynatreeMeasurement(
        day=s.day.value, tree=s.tree.value, measurement=s.measurement.value, measurement_type=s.method.value)
    
    if "Elasto" in probe.value:
        df = data_obj.data_pulling
        df = df[["Elasto(90)"]]
    else: 
        df = data_obj.data_optics
        mask = [i for i in df.columns if "Y0" in i[1]]
        df = df.loc[:,mask]
        df.columns = [i[0] for i in df.columns]
        df = df - df.iloc[0,:]
        df = df[probe.value]
    # solara.DataFrame(df)
    fig = px.scatter(df, height = s.height.value, width=s.width.value,
                          title=f"Dataset: {s.method.value}, {s.day.value}, {s.tree.value}, {s.measurement.value}",  
                          **kwds)
    solara.FigurePlotly(fig, on_click=set_click_data)

    return df

kwds = {"template": "plotly_white", 
        }

probe = solara.reactive("Elasto")
choices_disabled = solara.reactive(False)
t_from = solara.reactive(0)
t_to = solara.reactive(0)

@solara.component
def Page():
    solara.Title("DYNATREE: FFT")
    solara.Style(s.styles_css)
    with solara.Sidebar():
        s.Selection(exclude_M01=True, optics_switch=False)
        s.ImageSizes()

    if s.measurement.value not in s.available_measurements(s.df.value, s.day.value, s.tree.value, s.method.value, exclude_M01=True):
        print(f"Mereni {s.measurement.value} neni k dispozici, koncim")
        return
    with solara.lab.Tabs():
        with solara.lab.Tab("FFT"):
            try:
                DoFFT()
            except:
                with solara.Error():
                    solara.Markdown(
"""
**Bohužel nastala nějaká chyba.**

* Zkus nahlásit při jaké činnosti a při jaké volbě měření a sledovaných veličin. 
* Možná jsou špatné meze.
""")
        with solara.lab.Tab("Návod"):
            Navod()

def ChooseProbe():
    data_obj = lib_dynatree.DynatreeMeasurement(
        day=s.day.value, tree=s.tree.value, measurement=s.measurement.value, measurement_type=s.method.value)
    probes = ["Elasto"] + ["Pt3","Pt4"] + [f"BL{i}" for i in range(44,68)] 
    if not data_obj.is_optics_available:
        probe.value = ["Elasto"]
    # solara.Text(str(probe.value))
    solara.ToggleButtonsMultiple(value=probe, values=probes)    
    return data_obj

df_limits = solara.reactive(pd.DataFrame())

def save_limits():
    ans = {
            'measurement_type': [s.method.value],
            'day':[s.day.value], 
            'tree':[s.tree.value], 
            'measurement':[s.measurement.value], 
            'probe':[probe.value[0]],
            'from':[t_from.value],
            'to':[t_to.value],
        }
    ans = pd.DataFrame(ans)
    df_limits.value = pd.concat([ans,df_limits.value]).reset_index(drop=True)
    

def DoFFT():
    data_obj = ChooseProbe()
    if "Elasto" in probe.value:
        probe.value = ["Elasto"]
    df = plot()
    with solara.Row():
        with solara.Tooltip("Hodnoty je možné zadat číslem do políčka nebo kliknutím na bod v grafu výše. Se shiftem se nastavuje konec časového intervalu."):
            with solara.Column():
                solara.Markdown("**Limits for FFTⓘ:**")
        solara.InputFloat("From",value=t_from)
        solara.InputFloat("To",value=t_to)
        solara.Button(label="Save to table", on_click=save_limits)
    if (t_to.value == 0) or (t_to.value < t_from.value): 
        subdf = df.interpolate(method='index').loc[t_from.value:,:]
    else:
        t_final = t_to.value
        subdf = df.interpolate(method='index').loc[t_from.value:t_final,:]

    # Find new dataframe, resampled and restricted
    oldindex = subdf.index
    newindex = np.arange(oldindex[0],oldindex[-1], DT)
    newdf = pd.DataFrame(index=newindex, columns=subdf.columns)
    for i in subdf.columns:
        newdf[i] = np.interp(newindex, oldindex, subdf[i].values)
    # solara.display(newdf.head())
    fig = px.scatter(newdf, height = s.height.value, width=s.width.value,
                          title=f"Dataset: {s.method.value}, {s.day.value}, {s.tree.value}, {s.measurement.value}<br>Detail from {newdf.index[0]:.2f} to {newdf.index[-1]:.2f} resampled with dt={DT}",
                          **kwds)
    solara.FigurePlotly(fig)    

    # get FFT output
    time_fft = newdf.index.values    
    N = time_fft.shape[0]  # get the number of points
    xf_r = fftfreq(N, DT)[:N//2]
    df_fft = pd.DataFrame(index=xf_r, columns=newdf.columns)
    for col in newdf.columns:
        signal_fft = newdf[col].values
        time_fft = time_fft - time_fft[0]
        signal_fft = signal_fft - np.nanmean(signal_fft) # mean value to zero
        yf = fft(signal_fft)  # preform FFT analysis
        yf_r = 2.0/N * np.abs(yf[0:N//2])
        df_fft[col] = yf_r

    fig = px.scatter(df_fft, height = s.height.value, width=s.width.value,
                          title=f"FFT spectrum for {s.method.value}, {s.day.value}, {s.tree.value}, {s.measurement.value}<br>Limits: from {newdf.index[0]:.2f} to {newdf.index[-1]:.2f}", range_x=[0,3], log_y=True, range_y=[0.001,100],  
                          **kwds)
    fig.update_layout(xaxis_title="Freq/Hz", yaxis_title="FFT amplitude")
    solara.FigurePlotly(fig)    

    # show saved data    
    solara.display(df_limits.value)
    with solara.Row():
        solara.Button(label="Drop the first row", on_click=first_drop)
        solara.FileDownload(df_limits.value.to_csv(), filename=f"limits_for_FFT.csv", label="Download as csv")

def first_drop():
    df_limits.value = df_limits.value.iloc[1:,:].reset_index(drop=True)
   
@solara.component
def Navod():
    solara.Markdown(
"""
**TL;DR**

* Klikáním na tlačítka vyber zdroj dat a která data chceš zkoumat. 
* Je možné vybrat více voleb. 
* Volba Elasto eliminuje všecny ostatní případné volby.
* Nastav časový interval na kterém chceš dělat FFT zapsáním hodnot do políček nebo kliknutím na bod v grafu (koncový bod se shiftem).

**Postup**

* Měření vyber v levém panelu. V horním menu vyber sledované veličiny.
* Pokud není k dispozici optika, použije se vždy "Elasto", extenzometr. 
* Pokud vybereš "Elasto", použijí se data z extenzometru a další volby se ignorují. Jinak
  se zpracovává podle optiky vše zatržené.
* Pokud není optika k dispozici, bere se pro FFT automaticky extenzometr, tj. Elasto.
* Vše se interpoluje na 0.01s, to je v souladu se vzorkovaci frekvenci optiky.
* Podle grafu můžeš vybrat rozsah pro FFT. Začátek a konec se zapisuje do políček pod grafem. 
* Nulová horní mez znamená rozsah až do konce.
* BL jsou konce probů typu BendLine. Všechny výchylky jsou brány v ose y a je uvažována změna oproti výchozímu stavu, tj. uvažujeme posunutí, ne absolutní souřadnice v prostoru.


**Poznámky**

* Extenzometr se bere přímo z měření. Není tedy synchronizován s optikou a pokud jsou 
  k dispozici oba údaje, mohou se časy maličko lišit posunutím.
"""
        )


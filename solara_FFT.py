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

import logging
logger = logging.getLogger("Solara_FFT")
logger.setLevel(logging.DEBUG)

DT = 0.01
pd.set_option('display.max_rows', 500)

def set_click_data(x=None):
    if x['device_state']['shift']:
        t_to.value = x['points']['xs'][0]
    else:
        t_from.value = x['points']['xs'][0]

def save_freq_on_click(x=None):
    logger.debug(f"FFT clicked. Event: {x}")
    logger.debug(f"Previous value: {fft_freq.value}")
    if pd.isna(fft_freq.value):
        fft_freq.set(f" {x['points']['xs'][0]:.4f}")
    else:
        fft_freq.set(fft_freq.value + f" {x['points']['xs'][0]:.4f}")
    logger.debug(f"Current value: {fft_freq.value}")

def plot():
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
remark = solara.reactive("")
# peaks = solara.reactive("")
fft_freq = solara.reactive("")

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
    preload_data()
    with solara.lab.Tabs():
        with solara.lab.Tab("FFT"):
            try:
                DoFFT()
                ShowSavedData()
            except:
                with solara.Error():
                    solara.Markdown(
"""
**Bohužel nastala nějaká chyba.**

* Zkus nahlásit při jaké činnosti a při jaké volbě měření a sledovaných veličin. 
* Možná jsou špatné meze. Je určitě dolní mez menší než horní?
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

df_limits = solara.reactive(pd.read_csv("csv/solara_FFT.csv", index_col=[0,1,2,3,4], dtype={'probe':str}))

def save_limits():
    if len(probe.value) == 0:
        solara.lab.ConfirmationDialog(True, content="Select at least one variable.")
        return
    df_limits.value.loc[(
        s.method.value, s.day.value, s.tree.value, s.measurement.value, probe.value[0]),:
        ] = [t_from.value, t_to.value, fft_freq.value, remark.value]
    df_limits.value = df_limits.value.sort_index()
    # print(df_limits.value)
    
def preload_data():
    logger.debug("preload data started")
    logger.debug(f"looking for {s.method.value} {s.day.value} {s.tree.value} {s.measurement.value} {probe.value[0]}")
    coordinates = (s.method.value, s.day.value, s.tree.value, s.measurement.value,probe.value[0])
    # breakpoint()
    test = coordinates in df_limits.value.index
    # breakpoint()
    if test:
        row = df_limits.value.loc[coordinates,:]
        # breakpoint()
        t_from.value = row['from']
        t_to.value = row['to']
        remark.value = row['remark']
        fft_freq.value = row['peaks']
        logger.debug("preload data have been used")
    else:
        t_from.value = 0
        t_to.value = 0
        remark.value = ""
        fft_freq.value = ""
        logger.debug("preload data have NOT been used, using defaults")

@solara.component
def FFT_parameters():
    with solara.Card():
        with solara.Row():
            with solara.Tooltip("Hodnoty je možné zadat číslem do políčka nebo kliknutím na bod v grafu výše. Se shiftem se nastavuje konec časového intervalu."):
                with solara.Column():
                    solara.Markdown("**Limits for FFTⓘ:**")
            solara.InputFloat("From",value=t_from)
            solara.InputFloat("To",value=t_to)
        with solara.Row():
            solara.InputText("Remark", value=remark)
            solara.Button(label="Save to table", on_click=save_limits)


@solara.component
def DoFFT():
    logger.debug("DoFFT entered")
    data_obj = ChooseProbe()
    if "Elasto" in probe.value:
        probe.value = ["Elasto"]
    df = plot()

    FFT_parameters()
    if pd.isna(t_to.value):
        t_to.value = 0
    if pd.isna(t_from.value):
        t_from.value = 0
    if (t_to.value == 0) or (t_to.value < t_from.value): 
        subdf = df.interpolate(method='index').loc[t_from.value:,:]
    else:
        t_final = t_to.value
        subdf = df.interpolate(method='index').loc[t_from.value:t_final,:]
    
    # Find new dataframe, resampled and restricted
    oldindex = subdf.index
    if len(oldindex) < 1:
        return
    # breakpoint()
    newindex = np.arange(oldindex[0],oldindex[-1], DT)
    newdf = pd.DataFrame(index=newindex, columns=subdf.columns)
    for i in subdf.columns:
        newdf[i] = np.interp(newindex, oldindex, subdf[i].values)
    # solara.display(newdf.head())
    fig = px.scatter(newdf, height = s.height.value, width=s.width.value,
                          title=f"Dataset: {s.method.value}, {s.day.value}, {s.tree.value}, {s.measurement.value}<br>Detail from {newdf.index[0]:.2f} to {newdf.index[-1]:.2f} resampled with dt={DT}",
                          **kwds)
    solara.FigurePlotly(fig)    

    ShowFFTdata()
        
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

    figFFT = px.scatter(df_fft, height = s.height.value, width=s.width.value,
                          title=f"FFT spectrum for {s.method.value}, {s.day.value}, {s.tree.value}, {s.measurement.value}<br>Limits: from {newdf.index[0]:.2f} to {newdf.index[-1]:.2f}", range_x=[0,3], log_y=True, range_y=[0.001,100],  
                          **kwds)
    figFFT.update_layout(xaxis_title="Freq/Hz", yaxis_title="FFT amplitude")
    solara.FigurePlotly(figFFT, on_click=save_freq_on_click)    

@solara.component
def ShowFFTdata():
    logger.debug(f"ShowFFTdata entered {fft_freq.value}")
    with solara.Row():
        solara.Markdown(f"**FFT freq**: {fft_freq.value}")
        solara.Button(label="Erase", on_click=smazat_fft)
        solara.Button(label="Save to table", on_click=save_limits)


def smazat_fft(x=None):
    fft_freq.value = ""
    
# filter_method = solara.reactive(False)
filter_day = solara.reactive(False)
filter_tree = solara.reactive(False)
filter_probe = solara.reactive(False)
# filtered_df = solara.reactive(pd.DataFrame())

@solara.component
def ShowSavedData():
    # show saved data    
    with solara.Card():
        solara.Markdown("**Table with data**")
        with solara.Card():
            solara.Markdown("**Data restrictions**")
            with solara.Row():
                solara.Switch(label=f"Day {s.day.value}", value=filter_day)    
                solara.Switch(label=f"Tree {s.tree.value}", value=filter_tree)    
                solara.Switch(label=f"Probe {probe.value[0]}", value=filter_probe)    
        logger.debug(f"ShowSavedData entered {filter_day.value} {filter_tree.value} {filter_probe.value}")
        try:
            filtered_df = df_limits.value.copy()
            if filter_day.value:
                filtered_df = filtered_df.loc[
                    (slice(None), s.day.value,slice(None),slice(None),slice(None)), :]
            if filter_tree.value:
                filtered_df = filtered_df.loc[
                    (slice(None), slice(None), s.tree.value,slice(None),slice(None)), :]
            if filter_probe.value:
                filtered_df = filtered_df.loc[
                    (slice(None), slice(None),slice(None),slice(None),probe.value[0]), :]
            # df_limits.value = tempdf
            solara.display(filtered_df)
        except:
            logger.error(f"ShowSavedData failed")
        with solara.Row():
            solara.Button(label="Save current to table", on_click=save_limits)
            solara.FileDownload(df_limits.value.to_csv(), filename=f"limits_for_FFT.csv", label="Download as csv")

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


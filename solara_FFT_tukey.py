#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 23:33:06 2024

@author: marik
"""

import solara
import solara_select_source as s
import lib_dynatree
from solara.lab import task
import matplotlib.pyplot as plt
# from lib_dynasignal import do_fft_image, process_signal, do_welch_image
import lib_FFT
import pandas as pd
import numpy as np
import plotly.express as px


# https://stackoverflow.com/questions/37470734/matplotlib-giving-error-overflowerror-in-draw-path-exceeded-cell-block-limit
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

df_failed = pd.read_csv("csv/FFT_failed.csv").values.tolist()
df_fft_long = pd.read_csv("../outputs/FFT_csv_tukey.csv")
df_fft_all = df_fft_long.pivot(
    index = ["type","day","tree","measurement"],
    values="peak",
    columns="probe")

button_color = solara.reactive('primary')
probe = solara.reactive("Elasto(90)")
restrict = 50 # cut the FFT at 50Hz


def ChooseProbe():
    data_obj = lib_dynatree.DynatreeMeasurement(
        day=s.day.value, tree=s.tree.value, measurement=s.measurement.value, measurement_type=s.method.value)
    probes = ["Elasto(90)", "blueMaj", "yellowMaj", "Pt3","Pt4", 'a01_z', 'a02_z', 'a03_z', 'a04_z']
    with solara.Column():
        solara.ToggleButtonsSingle(value=probe, values=probes, on_value=nakresli_signal)
        test_is_failed = [s.method.value, s.day.value, s.tree.value, s.measurement.value, probe.value
                     ] in df_failed
        if test_is_failed:
            solara.Warning(f"Active probe: {probe.value}. This measurement is classified as failed.")
        else:
            solara.Info(f"Active probe: {probe.value}. The measurement is classifid as suitable for processing.")
    # solara.ToggleButtonsMultiple(value=probe, values=probes, mandatory=True)
    return data_obj

def resetujmethod(x=None):
    s.get_measurements_list()
    resetuj()
    
def resetuj(x=None):
    # Srovnani(resetuj=True)
    s.measurement.set(s.measurements.value[0])
    nakresli_signal()
    
def set_button_color(x=None):
    button_color.value = 'primary'    

def zpracuj(x=None):
    m = lib_dynatree.DynatreeMeasurement(day=s.day.value, 
        tree=s.tree.value, 
        measurement=s.measurement.value, 
        measurement_type=s.method.value)
    probename = probe.value
    release_source = probename
    if probe.value in ["blueMaj","yellowMaj"]:
        probe_final = m.identify_major_minor[probe.value]
        release_source="Elasto(90)"
    else:
        probe_final = probe.value
    sig = lib_FFT.DynatreeSignal(m, probe_final, release_source=release_source)

    return {'main_peak': sig.main_peak, 'signal':sig.signal, 'fft':sig.fft, 'signal_full':sig.signal_full}

@task
def nakresli_signal(x=None):
    output = zpracuj()
    
    fig, ax = plt.subplots(2,1,figsize=(10,5))
    tempsignal = output['signal_full']
    tempsignal = tempsignal.dropna()
    tempsignal = tempsignal - tempsignal.iloc[0]
    tempsignal.plot(ax=ax[0])
    output['signal'].plot(ax=ax[0])
    limits = (output['signal'].min(), output['signal'].max())
    ax[0].set (title=f"{s.method.value} {s.day.value} {s.tree.value} {s.measurement.value} {probe.value}",
            ylim=limits)
    output['fft'].plot(logy=True, xlim=(0,10), ax=ax[1])
    ax[0].grid()
    ax[1].grid()
    ymax = output['fft'].max()
    ax[1].set(ylim=(ymax/10**4,ymax*2), xlabel="Freq / Hz", ylabel="Amplitude")
    ax[0].set(xlim=(0,None), xlabel="Time / s", ylabel="Value")
    test_is_failed = [s.method.value, s.day.value, s.tree.value, s.measurement.value, probe.value
                     ] in df_failed    
    if test_is_failed:
        value = np.nan
    else:
        value = output['main_peak']
        ax[1].axvline(value, color='r', linestyle="--")
    plt.tight_layout()
    return (fig)    
    # plt.close('all')

@solara.component
def Page():
    solara.Title("DYNATREE: FFT s automatickou detekci vypuštění")
    solara.Style(s.styles_css)
    with solara.Sidebar():
        s.Selection(exclude_M01=True, 
                    optics_switch=False, 
                    day_action = resetuj,
                    tree_action = resetuj,
                    measurement_action = nakresli_signal, 
                    )  
        s.ImageSizes()

        with solara.Column(align='center'):
            solara.Button(label='Redraw',on_click=nakresli_signal, color='primary')

    ChooseProbe()
    
    with solara.lab.Tabs():
        with solara.lab.Tab("Static image"):
            try:
                solara.ProgressLinear(nakresli_signal.pending)
                if nakresli_signal.not_called:
                    nakresli_signal()
                if nakresli_signal.finished:
                        solara.FigureMatplotlib(nakresli_signal.value)
                        plt.close('all')
            except:
                pass
        with solara.lab.Tab("Dynamic FFT image"):
            try:
                data = zpracuj()
                df_fft = data['fft'].loc[:restrict]
                if isinstance(df_fft.name, tuple):
                    df_fft.name = df_fft.name[0]
                ymax = df_fft.to_numpy().max()
                figFFT = px.line(df_fft, 
                                 height = s.height.value, width=s.width.value,
                                 title=f"FFT spectrum: {s.method.value}, {s.day.value}, {s.tree.value}, {s.measurement.value}, {probe.value}", 
                                 log_y=True, range_x=[0,10], range_y=[ymax/100000, ymax*2]
                )
                figFFT.update_layout(xaxis_title="Freq/Hz", yaxis_title="FFT amplitude")
                solara.FigurePlotly(figFFT)
            except:
                pass
        with solara.lab.Tab("Statistiky"):
            # breakpoint()
            subdf = df_fft_all.loc[(s.method.value,s.day.value,s.tree.value,slice(None)),:]
            solara.display(subdf)
        with solara.lab.Tab("Popis"):
            solara.Markdown(
f"""
# Co tu je za data?

* Signál je dointerpolovaný na 100Hz pro elasto, inclino a optiku (optika občas vypadávala a elasto/inclino mají menší frekvenci) 
a ponechaný na 5000Hz pro akcelerometry.
* Podle maxima signálu se určí okamžiku vypuštění. Od něj se bere 60 sekund. Pokud
signál netrvá tak dlouho, doplní se nulami. 
* Na signál se aplikuje tukey okénko pro oříznutí okrajových efektů.
* Výsledný signál se protáhne přes FFT.

# Která data jsou označena jako nevalidní?

* Pokud má signál záznam v souboru `csv/FFT_failed.csv`, označí se jako špatný a nezpracovává
se do výsledných statistik. 
* Pokud chceš vynechat tohle měření, přidej do souboru tento řádek:
    
        {s.method.value},{s.day.value},{s.tree.value},{s.measurement.value},{probe.value}
        
* Pokud chceš projít a vychat více nebo hodně obrázků, je efektivnější si stáhout
obrázky fft z ERC disku, projít je, odmazávat co se nehodí, potom koši najít jména
odmazaných souborů a ta jednoduchým najdi nahraď přetransformovat na řádky co csv souboru.


# Ovládání

* Vyber měření a probe. Jedna záložka ukazuje statický obrázek signál a fft, druhá záložka
dynamický obrázek s fft.
* Pokud chceš plný rozsah fft, použij tlačítko autoscale u obrázku a poté si vyber, co potřebuješ.
* V dynamickém obrázku je kvůli lepší odezvě použit jenom rozsah do 50Hz.

"""                
                )
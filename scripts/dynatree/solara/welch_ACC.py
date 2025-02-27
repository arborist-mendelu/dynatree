#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 23:33:06 2024

@author: marik
"""

import solara
import dynatree.solara.select_source as s
from solara.lab import task
import matplotlib.pyplot as plt
from dynatree.dynasignal import do_welch
import pandas as pd
import plotly.express as px
import dynatree.dynatree as dynatree
import dynatree.FFT as FFT
import time
import solara_auth

# https://stackoverflow.com/questions/37470734/matplotlib-giving-error-overflowerror-in-draw-path-exceeded-cell-block-limit
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

dynatree.logger.setLevel(dynatree.logger_level)

__start = time.time()

acc_fft_axis = solara.reactive('a03_z')
tukey_alpha = solara.reactive(0.1)
tab_value = solara.reactive(1)
button_color = solara.reactive('primary')

def resetuj(x=None):
    # Srovnani(resetuj=True)
    s.measurement.set(s.measurements.value[0])
    nakresli()
    
def set_button_color(x=None):
    button_color.value = 'primary'    

n = solara.reactive(8)

@task
def nakresli(x=None):
    m = dynatree.DynatreeMeasurement(day=s.day.value,
                                         tree=s.tree.value,
                                         measurement=s.measurement.value,
                                         measurement_type=s.method.value)
    title = (f"{s.method.value} {s.day.value} {s.tree.value} {s.measurement.value}")
    axis = acc_fft_axis.value
    title = title + ": " + acc_fft_axis.value
    sig = FFT.DynatreeSignal(m, axis)

    fig, ax = plt.subplots()
    sig.signal_full.plot(ax=ax)
    sig.signal.plot(ax=ax)
    ax.set(ylim=(sig.signal.min(), sig.signal.max()))
    ax.set(title = title + ": " + acc_fft_axis.value)
    # plt.tight_layout()
    ans = do_welch(pd.DataFrame(sig.signal),  nperseg=2**n.value)

    figB, axB = plt.subplots(figsize=(10,5))
    ans.plot(ax=axB)
    axB.set(title = title, xlabel='Freq / Hz', ylabel="Power spectral density")
    axB.grid()
    axB.set(yscale='log')

    figB = px.line(ans, height = 400, width=1200,
                          title=f"Power spectrum "+title, 
                          log_y=True, #range_x=[0,50], 
                          # range_y=[ymax/10000, ymax*2]
    )
    figB.update_layout(xaxis_title="Freq/Hz", yaxis_title="Power")    

    
    button_color.value = None
    return fig, figB

@solara.component
def Page():
    if not solara_auth.user.value:
        solara_auth.LoginForm()
        return
    solara.Title("DYNATREE: Welch for ACC")
    solara.Style(s.styles_css)
    with solara.Sidebar():
        s.Selection(exclude_M01=True, 
                    optics_switch=False,       
                    day_action = resetuj,
                    tree_action = resetuj,
                    measurement_action = nakresli, 
                    )  
        with solara.Column(align='center'):
            solara.Button(label='Redraw',on_click=nakresli, color='primary')

    # solara.ToggleButtonsMultiple(
    solara.ToggleButtonsSingle(
        value=acc_fft_axis, values=[
            'a01_x', 'a01_y', 'a01_z', 
            'a02_x', 'a02_y', 'a02_z', 
            'a03_x', 'a03_y', 'a03_z', 
            'a04_x', 'a04_y', 'a04_z'], 
        mandatory=True, 
        on_value=set_button_color
        )

    with solara.Row():    
        solara.Button(label='Redraw',on_click=nakresli, color=button_color.value)

    # try:
        
    solara.ProgressLinear(nakresli.pending) 

    if nakresli.finished:
        fig, figB = nakresli.value

    with solara.lab.Tabs(value=tab_value):
        with solara.lab.Tab("Time domain"):
            if nakresli.finished:
                solara.FigureMatplotlib(fig, format='png')            
        # with solara.lab.Tab("Tukey window"):
        #     if nakresli.finished:
        #         # solara.Text(f" blue: original signal, orange: windowed signal")
        #         solara.FigureMatplotlib(figB, format='png')                   
        with solara.lab.Tab("Welch"):
            if nakresli.finished:
                # solara.FigureMatplotlib(figB, format='png')
                # with solara.Column(align='left'):
                with solara.Row():
                    solara.Markdown(r"$n$ (where $\text{nperseg}=2^n$)")
                    solara.ToggleButtonsSingle(values=list(range(6,13)), value=n, on_value=nakresli)
                    
                solara.FigurePlotly(figB)
                # solara.FigurePlotly(figD)
        with solara.lab.Tab("Popis"):
            solara.Markdown(
"""
* Pracuje se s původním vzorkováním 5000Hz.
* Time domain: časový průběh zvoleného signálu. Oranžová tečka je automatem určené vypuštění. 
  Toto je stanoveno jako maximum na intervalu od 25 sekund do konce.
* Tukey: Ořezaný signál od vypuštění na délku 60 sekund. Pokud je kratší, 
  je doplněno nulami. Poté je aplikováno Tukey okénko na celou minutu.
* Welch: Interaktivní graf, je možné zoomovat. Pokud nevyhovuje defualt nastavení, 
  odzoomuj a znovu vyber, co tě zajímá.
"""                
                )


    if nakresli.not_called:
        nakresli()         
    # except:
        # pass

    plt.close('all')

dynatree.logger.info(f"welch_ACC.py loaded in {time.time()-__start}")
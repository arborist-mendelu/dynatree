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
from lib_dynasignal import do_fft_image, process_signal, do_welch_image

# https://stackoverflow.com/questions/37470734/matplotlib-giving-error-overflowerror-in-draw-path-exceeded-cell-block-limit
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000


acc_fft_axis = solara.reactive('a03_z')
acc_alpha = solara.reactive(0.1)
tab_value = solara.reactive(1)
button_color = solara.reactive('primary')

def resetuj(x=None):
    # Srovnani(resetuj=True)
    s.measurement.set(s.measurements.value[0])
    nakresli()
    
def set_button_color(x=None):
    button_color.value = 'primary'    

@task
def nakresli(x=None):
    m = lib_dynatree.DynatreeMeasurement(day=s.day.value, 
        tree=s.tree.value, 
        measurement=s.measurement.value, 
        measurement_type=s.method.value)
    title = (f"{s.method.value} {s.day.value} {s.tree.value} {s.measurement.value}")
    axis = acc_fft_axis.value
    a = m.data_acc5000.loc[:,axis].copy()
    a.loc[:20] = 0
    release = a.idxmax()
    fig, ax = plt.subplots(#2,1, sharey=True, 
                           figsize=(12,4))
    # m.data_acc.loc[:,axis].plot(ax = ax[0], ylabel="100Hz")
    m.data_acc5000.loc[:,axis].plot(ax = ax, ylabel="5000Hz")
    yrange = (
        m.data_acc.loc[release+2:,axis].min(), 
        m.data_acc.loc[release+2:,axis].max())
    ax.set(ylim=yrange)
    ax.plot([release],[0],"o")
    # ax[1].plot([release],[0],"o")
    ax.legend([axis] + ["release autodetection"])
    plt.suptitle(title + ": " + acc_fft_axis.value)
    # plt.tight_layout()
    

    figB, axB = plt.subplots(#2,1 sharex=True,
                             figsize=(12,4),)
    signal_5000 = m.data_acc5000.loc[release:,[axis]]
    signal_5000.plot(style='.', ms=1, ax=axB)
    signal_5000 = process_signal(signal_5000, start=release, tukey=acc_alpha.value)
    signal_5000.plot(style='.', ms=1, ax=axB)
    # signal_100 = m.data_acc.loc[release:,axis]
    # signal_100.plot(style='.', ms=1, ax=axB[0])
    # signal_100 = process_signal(signal_100, start=release, tukey=acc_alpha.value, dt=0.01)
    # signal_100.plot(style='.', ms=1, ax=axB[0])
    ylim = (-.5,.5)
    ylim = yrange
    axB.set(ylim=ylim, ylabel="5000Hz")
    # axB[0].set(ylim=ylim, ylabel="100Hz")
    axB.legend(["original signal", "1 min signal with tukey window"])
    plt.suptitle(title + ": " + acc_fft_axis.value)
    # plt.tight_layout()

    # figC = None
    figD = None
    # figC = do_fft_image(signal_100, 0.01, "Signal downsampled. "+title, restrict=50)['fig']
    figC = do_fft_image(signal_5000, 0.0002, "Signal original. "+title, restrict=50)['fig']
    # figD = do_welch_image(signal_5000, "Signal original. "+title, restrict=50,  nperseg=2**15)['fig']
    
    
    button_color.value = None
    return fig, figB, figC, figD

@solara.component
def Page():
    solara.Title("DYNATREE: FFT srovnani ACC")
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
        solara.SliderFloat(label="Parameter of Tukey window", value=acc_alpha, min=0, max=1, step=0.01, on_value=nakresli)
        solara.Button(label='Redraw',on_click=nakresli, color=button_color.value)

    # try:
        
    solara.ProgressLinear(nakresli.pending) 

    if nakresli.finished:
        fig, figB, figC, figD = nakresli.value

    with solara.lab.Tabs(value=tab_value.value):
        with solara.lab.Tab("Time domain"):
            if nakresli.finished:
                solara.FigureMatplotlib(fig, format='png')            
        with solara.lab.Tab("Tukey window"):
            if nakresli.finished:
                # solara.Text(f" blue: original signal, orange: windowed signal")
                solara.FigureMatplotlib(figB, format='png')                   
        with solara.lab.Tab("FFT"):
            if nakresli.finished:
                # pass
                solara.FigurePlotly(figC)
                # solara.FigurePlotly(figD)
        with solara.lab.Tab("Popis"):
            solara.Markdown(
"""
* Pracuje se s původním vzorkováním 5000Hz.
* Time domain: časový průběh zvoleného signálu. Oranžová tečka je automatem určené vypuštění. Toto je stanoveno jako maximum na intervalu od 20 sekund do konce.
# * Tukey: Ořezaný signál od vypuštění na délku 60 sekund. Pokud je kratší, je vycenrováno a doplněno nulami. Poté je aplikováno Tukey okénko na celou minutu.
Parametr okénka je možné měnit.
* FFT: Interaktivní graf, je možné zoomovat. Pokud nevyhovuje defualt nastavení, odzoomuj a znovu vyber, co tě zajímá.
"""                
                )


    if nakresli.not_called:
        nakresli()         
    # except:
        # pass

    plt.close('all')
        

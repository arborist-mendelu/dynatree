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
import seaborn as sns
# import psutil
# import logging
import time

# lib_dynatree.logger.setLevel(logging.INFO)

# https://stackoverflow.com/questions/37470734/matplotlib-giving-error-overflowerror-in-draw-path-exceeded-cell-block-limit
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000


pd.options.display.float_format = '{:.3f}'.format
df_komentare = pd.read_csv("csv/FFT_comments.csv", index_col=[0,1,2,3,4])
df_failed = pd.read_csv("csv/FFT_failed.csv").values.tolist()
df_fft_long = pd.read_csv("../outputs/FFT_csv_tukey.csv")
df_fft_all = df_fft_long.pivot(
    index = ["type","day","tree","measurement"],
    values="peak",
    columns="probe")
df_fft_all = df_fft_all.loc[:,[
    'Elasto(90)', 'blueMaj', 'yellowMaj', 
    'Pt3', 'Pt4', 
    'a01_z', 'a02_z', 'a03_z', 'a04_z']]

button_color = solara.reactive('primary')
probe = solara.reactive("Elasto(90)")
restrict = 50 # cut the FFT at 50Hz
tab_value = solara.reactive(2)


def ChooseProbe():
    data_obj = lib_dynatree.DynatreeMeasurement(
        day=s.day.value, tree=s.tree.value, measurement=s.measurement.value, measurement_type=s.method.value)
    probes = ["Elasto(90)", "blueMaj", "yellowMaj", "Pt3","Pt4", 'a01_z', 'a02_z', 'a03_z', 'a04_z']
    with solara.Row():        
        solara.ToggleButtonsSingle(value=probe, values=probes, on_value=nakresli_signal)
        test_is_failed = [s.method.value, s.day.value, s.tree.value, s.measurement.value, probe.value
                     ] in df_failed
        if test_is_failed:
            solara.Error("Classified as failed.")
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

@lib_dynatree.timeit
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
@lib_dynatree.timeit
def nakresli_signal(x=None):
    output = zpracuj()
    
    plt.close('all')    
    f, ax = plt.subplots(2,1,figsize=(6,4))
    tempsignal = output['signal_full']
    tempsignal = tempsignal.dropna()
    tempsignal = tempsignal - tempsignal.iloc[0]
    tempsignal.plot(ax=ax[0])
    output['signal'].plot(ax=ax[0])
    limits = (output['signal'].min(), output['signal'].max())
    ax[0].set (title=f"{s.method.value} {s.day.value} {s.tree.value} {s.measurement.value} {probe.value}, {output['main_peak']:.04f}Hz",
            ylim=limits)
    output['fft'].plot(logy=True, xlim=(0,10), ax=ax[1])
    ax[0].grid()
    ax[1].grid()
    ymax = output['fft'].max()
    ax[1].set(ylim=(ymax/10**4,ymax*2), xlabel="Freq / Hz", ylabel="Amplitude")
    ax[1].yaxis.set_ticklabels([])
    ax[0].set(xlim=(0,None), xlabel="Time / s", ylabel="Value")
    test_is_failed = [s.method.value, s.day.value, s.tree.value, s.measurement.value, probe.value
                     ] in df_failed    
    if test_is_failed:
        value = np.nan
    else:
        value = output['main_peak']
        ax[1].axvline(value, color='r', linestyle="--")
    plt.tight_layout()
    return (f)    
    # plt.close('all')

# Funkce pro stylování - přidání hranice, když se změní hodnota v úrovni 'tree'
def add_horizontal_line(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    
    # Projdi všechny řádky a přidej stylování
    for i in range(1, len(df)):
        if df.index[i][1] != df.index[i - 1][1]:  # Pokud se změní 'tree'
            styles.iloc[i, :] = 'border-top: 2px solid black'  # Přidej hranici
    
    return styles

def ostyluj(subdf):
    cm = sns.light_palette("blue", as_cmap=True)
    subdf = (subdf.style.format(precision=3).background_gradient(#cmap=cm, 
                                             axis=None)
             .apply(add_horizontal_line, axis=None)
             .map(lambda x: 'color: lightgray' if pd.isnull(x) else '')
             .map(lambda x: 'background: transparent' if pd.isnull(x) else '')
             )
    return subdf

def myformat(df, column, row_index, value):
    if isinstance(value, float) and pd.isna(value):
        return "-"
    if isinstance(value, str):
        return value
    return f"{value:.3f}"

subdf = pd.DataFrame()
def on_action_cell(column, row_index):
    row = subdf.iloc[row_index,:]
    probe.value = column
    s.method.value = row['type']
    s.day.value = row['day']
    s.tree.value = row['tree']
    s.measurement.value = row['measurement']
    nakresli_signal()
    
cell_actions = [solara.CellAction(name="Show", on_click=on_action_cell)]

def FFT_remark():
    coords = (s.method.value, s.day.value, s.tree.value, s.measurement.value, probe.value)
    if coords in df_komentare.index:
        with solara.Column():
            with solara.Warning():
                solara.Text(df_komentare.loc[coords,:].iloc[0])    

def csv_line():
    solara.Markdown(
f"""
`{s.method.value},{s.day.value},{s.tree.value},{s.measurement.value},{probe.value}`
"""                        
        )

@solara.component
@lib_dynatree.timeit
def Page():
    global subdf
    initime = time.time()
    solara.Title("DYNATREE: FFT s automatickou detekci vypuštění a tukey oknem")
    solara.Style(s.styles_css)
    solara.Style("td {padding-left: 1em !important;}")
    with solara.Sidebar():
        if tab_value.value != 3:
            s.Selection(exclude_M01=True, 
                        optics_switch=False, 
                        day_action = resetuj,
                        tree_action = resetuj,
                        measurement_action = nakresli_signal, 
                        )  
            s.ImageSizes()
    
            with solara.Column(align='center'):
                solara.Button(label='Redraw',on_click=nakresli_signal, color='primary')
        # with solara.Card():
        #     with solara.Column():
        #         solara.Text(f'CPU: {psutil.cpu_percent(4)}%')
        #         solara.Text(f'Mem total: {psutil.virtual_memory()[0]/1000000000:.1f}GB')
        #         solara.Text(f'Mem used: {psutil.virtual_memory()[3]/1000000000:.1f}GB')
        #         solara.Text(f'Mem free: {psutil.virtual_memory()[4]/1000000000:.1f}GB')
                
        
    now = time.time()
    lib_dynatree.logger.info(f"Before choose probe after {now-initime}.")
    ChooseProbe()
    
    with solara.lab.Tabs(value=tab_value):
        with solara.lab.Tab("Static image"):
            if tab_value.value == 0:
                try:
                    solara.ProgressLinear(nakresli_signal.pending)
                    if nakresli_signal.not_called:
                        nakresli_signal()
                    if nakresli_signal.finished:
                        solara.FigureMatplotlib(nakresli_signal.value, format='png')
                    csv_line()
                    FFT_remark()
                    with solara.Info():
                        solara.Markdown(
"""
* Svislá červená čára je frekvence použitá do dalšího zpracování. Stanovena jako maximum na určitém frekvenčním 
  intervalu. Numerická hodnota je i v nadpisu obrázku.
* Pokud je toto měření pokažené, zkopíruj si řádek nad tímto rámečkem a přidá se mezi seznam zkažených.
"""                            
                            )
                except:
                    pass
        with solara.lab.Tab("Interactive FFT image"):
            if tab_value.value == 1:
                try:
                    solara.ProgressLinear(nakresli_signal.pending)
                    if nakresli_signal.finished:
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
        with solara.lab.Tab("Přehled barevně"):
            if tab_value.value == 2:
                with solara.Card(title=f"All days for tree {s.tree.value}"):
                    try:
                        subdfA = df_fft_all.loc[(slice(None),slice(None),s.tree.value,slice(None)),:]
                        subdfA = ostyluj(subdfA)
                        solara.display(subdfA)
                        # subdf = subdf.reset_index()
                        # solara.DataTable(subdf, items_per_page=100, format=myformat)
                    except:
                        pass
                with solara.Info():
                    solara.Markdown(
f"""
* Políčka jsou podbarvena podle hodnoty. V rámci jednoho dne (viz vodorovné čáry v tabulce) by měly být 
  barvy plus minus stejné.
* Krok mezi frekvencemi je 0.017Hz. Odchylka v mezích 0.02Hz nic neznamená, může se jednat o 
  vedlejší bod u širokého peaku.
* Vyjádření k některým měřením:
"""                        
                        )
                    solara.display(df_komentare)
        with solara.lab.Tab("Přehled s odkazy"):
            if tab_value.value == 3:
                with solara.Card(title=f"All days for tree {s.tree.value}"):
                    try:
                        subdf = df_fft_all.loc[(slice(None),slice(None),s.tree.value,slice(None)),:]
                        # subdf = ostyluj(subdf)
                        # solara.display(subdf)
                        subdf = subdf.reset_index()
                        solara.DataTable(subdf, items_per_page=100, format=myformat,cell_actions=cell_actions)                        
                        solara.HTML(tag="script", unsafe_innerHTML=
"""
function applyGradient() {
  const rows = document.querySelectorAll('tr');
  const values = [];
  // Najdi číselné hodnoty v atributech title buněk od pátého sloupce dál
  rows.forEach(row => {
    const cells = row.querySelectorAll('td:nth-child(n+5)');
    cells.forEach(cell => {
      const value = parseFloat(cell.getAttribute('title'));
      if (!isNaN(value)) {
        values.push(value);
      }
    });
  });
  // Zjisti minimum a maximum
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  // Aplikuj barvy na základě hodnot
  rows.forEach(row => {
    const cells = row.querySelectorAll('td:nth-child(n+5)');
    cells.forEach(cell => {
      const value = parseFloat(cell.getAttribute('title'));
      if (!isNaN(value)) {
        const intensity = (value - minValue) / (maxValue - minValue); // Normalizace hodnoty
        const colorIntensity = Math.floor(intensity * 255); // Přepočet na 0-255
        cell.style.backgroundColor = `rgb(${255 - colorIntensity}, ${255 - colorIntensity}, ${255})`;
      }
    });
  });
};

applyGradient();
"""
                            )
                    except:
                        pass
                with solara.Sidebar():
                    try:
                        csv_line()
                        solara.ProgressLinear(nakresli_signal.pending)
                        FFT_remark()
                        if nakresli_signal.finished:
                            solara.FigureMatplotlib(nakresli_signal.value)
                    except:
                        pass
                with solara.Info():
                    solara.Markdown(
"""
* Hodnoty v tabulce jsou zaokrouhlené. Co je jinde 0.69993 je zde jako 0.700. Pozor ať Tě to nezmate.
* Za položkama v tabulce jsou tři tečky, které po najetí umožní zobrazit statický graf a případnou 
  poznámku v sidebaru. Toto je možné použít na na data, která patří k existujícím měřením, ale mají
  v tabulce pomlčku, protože jsou tato měření vyhodnocena jako pokažená.
"""                        
                        )
        with solara.lab.Tab("Popis & Download"):
            with solara.Card(style={'background-color':"#FBFBFB"}):
                solara.Markdown("**Downloads**")
                with solara.Row(justify="space-around",style={'background-color':"#FBFBFB"}):
                    solara.FileDownload(df_fft_long.to_csv(), filename="fft_dynatree.csv", label="Peaks in long format")
                    solara.FileDownload(df_fft_all.to_csv(), filename="fft_dynatree_wide.csv", label="Peaks in wide format")
                    solara.FileDownload(df_komentare.to_csv(), filename="FFT_comments.csv", label="Comments")
                    solara.FileDownload(pd.DataFrame(df_failed, columns=["type","day","tree","mesurement","probe"]).to_csv(index=None), filename="FFT_failed.csv", label="Failed")
                
            solara.Markdown(
f"""
# Co tu je za data?

* Signál je dointerpolovaný na 100Hz pro elasto, inclino a optiku (optika občas vypadávala a elasto/inclino mají menší frekvenci) 
  a ponechaný na 5000Hz pro akcelerometry.
* Podle maxima signálu se určí okamžiku vypuštění. Od něj se bere 60 sekund. Pokud
  signál netrvá tak dlouho, doplní se nulami. Místo maxima inklinometrů se bere maximum extenzometru 
  (chová se líp).
* Na signál se aplikuje tukey okénko pro oříznutí okrajových efektů.
* Výsledný signál se protáhne přes FFT.
* Záložky "Přehled barevně" a "Přehled s odkazy" ukazují hlavní frekvenci. Hodí se pro nalezení odlehlých měření. 
  Stejné hodnoty jsou podbarveny stejně. Hledáme odchylku od (v ideálním případě) jednobarevných bloků.
* Data s hodnotami peaků jsou ke stažení na odkazech nahoře.
* Data s evidencí nevalidních experimentů a data s komentáři jsou také ke stažení.

# Která data jsou označena jako nevalidní?

* Pokud má signál záznam v souboru `csv/FFT_failed.csv`, označí se jako špatný a nezpracovává
  se do výsledných statistik. Generuje se průběh signálu a výstup FFT, ten je <a href='/static/public/FFT_spectra.zip'>ke stažení zde</a>.
* Pokud chceš vynechat tohle měření, přidej do souboru tento řádek:
    
        {s.method.value},{s.day.value},{s.tree.value},{s.measurement.value},{probe.value}
        
* Pokud chceš projít a vynechat více nebo hodně měření, je efektivnější si vygenerovat 
  offline obrázky (Robert). Postup: V souboru `lib_FFT.py` opravit volbu `plot='failed'` na
  `plot='all'`, spustit (pojede dlouho takže pomocí `nohup` nebo `screen`), stáhnout výstup 
  (`outputs/FFT_spectra.zip`), projít obrázky, odmazávat co se nehodí, 
  potom v koši najít jména odmazaných souborů a ta jednoduchým najdi 
  nahraď přetransformovat na řádky co csv souboru. Pro roztřídění do podadresářů podle stromů použij následující oneliner.

        for file in *_BK??_*; do dir="${{file#*_}}"; dir="${{dir%%_*}}"; mkdir -p "$dir"; mv "$file" "$dir/"; done

* Po úpravě csv souborů, kde jsou experimenty, které selhaly, je potřeba přegenerovat 
  soubor `FFT_csv_tukey.csv` obsahující všechny frekvence a distribuovat na servery. Kromě toho 
  rozkopírovat i csv soubory, které se měnily. Na um-bc201 takto:
  
    ```
    cd /mnt/zaloha/babice/dynatree-optika/
    conda activate dynatree
    snakemake
    rsync -zarv -P ../outputs jupyter.mendelu.cz:/babice/Mereni_Babice_zpracovani/
    rsync -zarv -P csv jupyter.mendelu.cz:/babice/Mereni_Babice_zpracovani/skripty/
    rsync -zarv -P csv_output jupyter.mendelu.cz:/babice/Mereni_Babice_zpracovani/skripty/
    ```

# Ovládání

* Vyber měření a probe. Jedna záložka ukazuje statický obrázek signál a fft, druhá záložka
  dynamický obrázek s fft.
* Pokud chceš plný rozsah fft, použij tlačítko autoscale u obrázku a poté si vyber, co potřebuješ.
* V dynamickém obrázku je kvůli lepší odezvě použit jenom rozsah do 50Hz.

# Komentáře

"""                
                )
            solara.display(df_komentare)
            

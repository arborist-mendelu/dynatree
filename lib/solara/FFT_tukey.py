#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 23:33:06 2024

@author: marik
"""

import solara
import lib.solara.select_source as s
import lib_dynatree
from solara.lab import task
import matplotlib.pyplot as plt
# from lib_dynasignal import do_fft_image, process_signal, do_welch_image
import lib_FFT
import pandas as pd
import numpy as np
import plotly.express as px
import solara.express as pxs
import seaborn as sns
# import psutil
# import logging
import time
import os
import config
from lib_dynasignal import do_welch
# from weasyprint import HTML, CSS
from pathlib import Path

# lib_dynatree.logger.setLevel(logging.INFO)

# https://stackoverflow.com/questions/37470734/matplotlib-giving-error-overflowerror-in-draw-path-exceeded-cell-block-limit
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000


import logging

logger = logging.getLogger("Solara_FFT")
logger.setLevel(logging.WARNING)

filelogger = logging.getLogger("FFT Rotating Log")
filelogger.setLevel(logging.INFO)
filehandler = logging.handlers.RotatingFileHandler(f"{os.path.expanduser('~')}/solara_log/solara_FFT.log", maxBytes=10000000, backupCount=10)
log_format = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s")
filehandler.setFormatter(log_format)
filelogger.addHandler(filehandler)


pd.options.display.float_format = '{:.3f}'.format
df_komentare = pd.read_csv(config.file['FFT_comments'], index_col=[0,1,2,3,4])
df_failed = pd.read_csv(config.file["FFT_failed"]).values.tolist()
df_fft_long = pd.read_csv(config.file["outputs/FFT_csv_tukey"])
df_fft_all = df_fft_long.pivot(
    index = ["type","day","tree","measurement"],
    values="peak",
    columns="probe")
df_fft_all = df_fft_all.loc[:,[
    'Elasto(90)', 'blueMaj', 'yellowMaj', 
    'Pt3', 'Pt4', 
    'a01_z', 'a02_z', 'a03_z', 'a04_z'
    ]]

button_color = solara.reactive('primary')
probe = solara.reactive("Elasto(90)")
restrict = 50 # cut the FFT at 50Hz
restrict = 5000
tab_value = solara.reactive(0)
subtab_value = solara.reactive(1)
manual_release_time = solara.reactive(0.0)
manual_end_time = solara.reactive(0.0)
n = solara.reactive(8)

def ChooseProbe():
    data_obj = lib_dynatree.DynatreeMeasurement(
        day=s.day.value, tree=s.tree.value, measurement=s.measurement.value, measurement_type=s.method.value)
    probes = ["Elasto(90)", "blueMaj", "yellowMaj", "Pt3","Pt4", 'a01_z', 'a02_z', 'a03_z', 'a04_z', 'a01_y', 'a02_y', 'a03_y', 'a04_y']
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
    manual_release_time.value = 0
    manual_end_time.value = 0
    fft_freq.value = load_fft_freq()
    s.measurement.set(s.measurements.value[0])
    nakresli_signal()
    
def set_button_color(x=None):
    button_color.value = 'primary'    

@lib_dynatree.timeit
def zpracuj(x=None, type='fft'):
    m = lib_dynatree.DynatreeMeasurement(day=s.day.value, 
        tree=s.tree.value, 
        measurement=s.measurement.value, 
        measurement_type=s.method.value)
    probename = probe.value
    release_source = probename
    if probe.value in ["blueMaj","yellowMaj"]:
        probe_final = m.identify_major_minor[probe.value]
        release_source = "Elasto(90)"
    else:
        probe_final = probe.value
    sig = lib_FFT.DynatreeSignal(m, probe_final, release_source=release_source)
    if sig.signal_full is None:
        return
    if manual_release_time.value > 0.0:
        sig.manual_release_time = manual_release_time.value
    if manual_end_time.value > 0.0:
        sig.signal_full = sig.signal_full[:manual_end_time.value]
    ans = {'main_peak': sig.main_peak, 'signal':sig.signal, 'signal_full':sig.signal_full}
    if type == 'fft':
        ans['fft'] = sig.fft
    if type == 'welch':
        ans['welch'] = sig.welch(nperseg=2**n.value)
    return ans

def spust_mereni(x=None):
    manual_release_time.value = 0
    manual_end_time.value = 0
    fft_freq.value = load_fft_freq()
    nakresli_signal()

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

fft_freq = solara.reactive("")
save_button_color = solara.reactive("none")

df_manual_peaks = solara.reactive(pd.read_csv(config.file["FFT_manual_peaks"], index_col=[0,1,2,3,4], dtype={'peaks':str}).fillna(""))
df_manual_peaks.value = df_manual_peaks.value.sort_index()

def save_peaks():
    df_manual_peaks.value.loc[(
        s.method.value, s.day.value, s.tree.value, s.measurement.value, probe.value),:
        ] = [fft_freq.value]
    df_manual_peaks.value = df_manual_peaks.value.sort_index()
    save_button_color.value = "none"
    filelogger.info(f"Saved {s.method.value},{s.day.value},{s.tree.value},{s.measurement.value},{probe.value},{fft_freq.value}")

def save_freq_on_click(x=None):
    logger.debug(f"FFT clicked. Event: {x}")
    logger.debug(f"Previous value: {fft_freq.value}")
    if pd.isna(fft_freq.value):
        fft_freq.set(f" {x['points']['xs'][0]:.4f}")
    else:
        fft_freq.set(fft_freq.value + f" {x['points']['xs'][0]:.4f}")
    logger.debug(f"Current value: {fft_freq.value}")
    save_button_color.value = "red"

def clear_fft_freq(x=None):
    fft_freq.value = ""
    save_button_color.value = "red"

def oprav_peaky(subdf):
    if not use_manual_peaks.value:
        return subdf
    idx = subdf.index
    for i,row in df_manual_peaks.value.iterrows():
        coords = tuple(row.name[:-1])
        probe = row.name[-1]
        value = float(row.iloc[0].strip().split(" ")[0])
        if not coords in idx:
            continue
        subdf.loc[coords,probe] = value
    return subdf

@solara.component
def SaveButton():
    solara.Button(label="Save to memory", on_click=save_peaks, color=save_button_color.value)

def load_fft_freq():
    coords = (s.method.value,s.day.value,s.tree.value,s.measurement.value,probe.value)
    if coords in df_manual_peaks.value.index:
        return df_manual_peaks.value.loc[coords,'peaks']
    return ""

use_manual_peaks = solara.reactive(True)

@solara.component
@lib_dynatree.timeit
def Page():
    global subdf

    initime = time.time()
    solara.Title("DYNATREE: FFT s automatickou detekci vypuštění a tukey oknem")
    solara.Style(s.styles_css)
    solara.Style("td {padding-left: 1em !important;} .widget-image {width: auto;}")
    with solara.Sidebar():
        if ((tab_value.value, subtab_value.value) != (1,1) ) & (tab_value.value !=2):
            if tab_value.value == 0:
                s.Selection(exclude_M01=True, 
                            optics_switch=False, 
                            day_action = resetuj,
                            tree_action = resetuj,
                            measurement_action = spust_mereni, 
                            )  
                s.ImageSizes()
                with solara.Column(align='center'):
                    solara.Button(label='Redraw',on_click=nakresli_signal, color='primary')
            else:
                s.Selection_trees_only()
    
        # with solara.Card():
        #     with solara.Column():
        #         solara.Text(f'CPU: {psutil.cpu_percent(4)}%')
        #         solara.Text(f'Mem total: {psutil.virtual_memory()[0]/1000000000:.1f}GB')
        #         solara.Text(f'Mem used: {psutil.virtual_memory()[3]/1000000000:.1f}GB')
        #         solara.Text(f'Mem free: {psutil.virtual_memory()[4]/1000000000:.1f}GB')
                
        
    now = time.time()
    lib_dynatree.logger.info(f"Before choose probe after {now-initime}.")
    if tab_value.value == 0:
        ChooseProbe()
    
    dark = {"background_color":"primary", "dark":True, "grow":True}

    with solara.lab.Tabs(value=tab_value, **dark):
        with solara.lab.Tab("Jedno měření", icon_name="mdi-chart-line"):
            with solara.lab.Tabs(value=subtab_value, **dark):
                with solara.lab.Tab("Time domain"):
                    solara.Markdown("# Time domain static explorer")
                    if (tab_value.value, subtab_value.value) == (0,0):
                        try:
                            solara.ProgressLinear(nakresli_signal.pending)
                            if nakresli_signal.not_called:
                                nakresli_signal()
                            if nakresli_signal.finished:
                                solara.FigureMatplotlib(nakresli_signal.value, format='png')
                                plt.close('all')
                            with solara.Row():
                                csv_line()
                                with solara.Tooltip("You may enter manual release time and click Redraw. Zero value is for automatical determination of release time."):
                                    solara.InputFloat(
                                        "release time", 
                                        value=manual_release_time)
                                with solara.Tooltip("You may enter manual end time and click Redraw. Zero value is for automatical determination (to the end, max 60 sec)."):
                                    solara.InputFloat(
                                        "end time", 
                                        value=manual_end_time)
                                solara.Button(label='Redraw',on_click=nakresli_signal, color='primary')
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
                with solara.lab.Tab("FFT (interactive)"):
                    solara.Markdown("# FFT interactive explorer")
                    if (tab_value.value, subtab_value.value) == (0,1):
                        # try:
                            solara.ProgressLinear(nakresli_signal.pending)
                            if nakresli_signal.not_called:
                                nakresli_signal()
                                # return
                            if nakresli_signal.finished:
                                data = zpracuj()
                                if data is None:
                                    return
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
                                solara.FigurePlotly(figFFT, on_click=save_freq_on_click)
                                with solara.Row():
                                    solara.Text(fft_freq.value)
                                    solara.Button(label="Clear", on_click=clear_fft_freq)
                                    SaveButton()
                                    solara.FileDownload(df_manual_peaks.value.to_csv(), filename=f"FFT_manual_peaks.csv", label="Download csv")
                                solara.display(df_manual_peaks.value)
                        # except:
                        #     pass
        
                with solara.lab.Tab("Welch (interactive)"):
                    solara.Markdown("# Welch spectrum interactive explorer")

                    if (tab_value.value, subtab_value.value) == (0,2):
                        try:
                            Welch_interactive()
                        except:
                            solara.Error("Něco se nepodařilo. Možná není dostupné měření.")

        with solara.lab.Tab("Jeden strom", icon_name="mdi-pine-tree"):
            with solara.lab.Tabs(value=subtab_value, **dark):
        
        
                with solara.lab.Tab("Přehled barevně"):
                    if (tab_value.value, subtab_value.value) == (1,0):
                        with solara.Card(title=f"All days for tree {s.tree.value}"):
                            solara.Switch(label="Use manual peaks (if any)", value=use_manual_peaks)
                            # try:
                            subdfA = df_fft_all.loc[(slice(None),slice(None),s.tree.value,slice(None)),:]
                            subdfA = oprav_peaky(subdfA.copy())
                            subdfA_copy = subdfA.copy()
                            subdfA = ostyluj(subdfA)
                            solara.display(subdfA)
                                # subdf = subdf.reset_index()
                                # solara.DataTable(subdf, items_per_page=100, format=myformat)
                            # except:
                                # pass                 
#                             css = CSS(string='''
# table {
#     transform: scale(0.7); 
#     transform-origin: top left; /* Nastaví počáteční bod transformace */
# }
#                                       ''')                 
                            with solara.Row():  
                                # solara.FileDownload(HTML(string=subdfA.to_html()).write_pdf(stylesheets=[css]), filename=f"dynatree-{s.tree.value}-table.pdf", label="Download PDF")
                                solara.FileDownload(subdfA.to_html(), filename=f"dynatree-{s.tree.value}-table.html", label="Download html")
                                solara.FileDownload(subdfA_copy.to_csv(), filename=f"dynatree-{s.tree.value}-table.csv", label="Download csv")
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
                    if (tab_value.value, subtab_value.value) == (1,1):
                        with solara.Sidebar():
                        # with solara.Columns():
                            with solara.Column():
                                try:
                                    with solara.Column():
                                        csv_line()
                                        solara.ProgressLinear(nakresli_signal.pending)
                                        FFT_remark()
                                        if nakresli_signal.finished:
                                            solara.FigureMatplotlib(nakresli_signal.value)
                                            plt.close('all')
                                            # data = zpracuj()
                                            # print("zpracovano")
                                            # df_fft = data['fft'].loc[:restrict]
                                            # if isinstance(df_fft.name, tuple):
                                            #     df_fft.name = df_fft.name[0]
                                            # ymax = df_fft.to_numpy().max()
                                            # figFFTB = px.line(df_fft, 
                                            #                  # height = "300px", width="400px",
                                            #                  title=f"FFT spectrum", 
                                            #                  log_y=True, range_x=[0,2], range_y=[ymax/100000, ymax*2]
                                            # )
                                            # figFFTB.update_layout(xaxis_title="Freq/Hz", yaxis_title="FFT amplitude")
                                            # solara.FigurePlotly(figFFTB)
                                    
                                except:
                                    pass
                        with solara.Card(title=f"All days for tree {s.tree.value}"):
                            # Does not work fine, the table is not updated.
                            # solara.Switch(label="Use manual peaks (if any)", value=use_manual_peaks)
                            try:
                                subdf = df_fft_all.loc[(slice(None),slice(None),s.tree.value,slice(None)),:]
                                # subdf = ostyluj(subdf)
                                # solara.display(subdf)
                                subdf = oprav_peaky(subdf.copy())
                                subdf = subdf.reset_index()
                                solara.DataTable(subdf, items_per_page=100, format=myformat,cell_actions=cell_actions)                        
        
                                with solara.Info():
                                    solara.Markdown(
                """
                * Hodnoty v tabulce jsou zaokrouhlené. Co je jinde 0.69993 je zde jako 0.700. Pozor ať Tě to nezmate.
                * Za položkama v tabulce jsou tři tečky, které po najetí umožní zobrazit statický graf a případnou 
                  poznámku v sidebaru. Toto je možné použít na na data, která patří k existujícím měřením, ale mají
                  v tabulce pomlčku, protože jsou tato měření vyhodnocena jako pokažená.
                """                        
                                        )
        
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


        with solara.lab.Tab("Popis & Download", icon_name="mdi-comment-outline"):
            if tab_value.value == 2:
                with solara.Sidebar():
                    with solara.Column():
                        with solara.Card():
                            solara.Markdown("**Downloads**")
                            solara.FileDownload(df_fft_long.to_csv(), filename="fft_dynatree.csv", label="Peaks in long format")
                            solara.FileDownload(df_fft_all.to_csv(), filename="fft_dynatree_wide.csv", label="Peaks in wide format")
                            solara.FileDownload(df_komentare.to_csv(), filename="FFT_comments.csv", label="Comments")
                            solara.FileDownload(pd.DataFrame(df_failed, columns=["type","day","tree","mesurement","probe"]).to_csv(index=None), filename="FFT_failed.csv", label="Failed")
                            solara.FileDownload(pd.read_csv(config.file["FFT_release"]).to_csv(index=None), filename="FFT_release.csv", label="Manual release times")
                            solara.FileDownload(pd.read_csv(config.file["FFT_manual_peaks"]).to_csv(index=None), filename="FFT_manual_peaks.csv", label="Manual peaks")
                
            
                
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
* Pokud automatika selhala při detekci vypuštění, je možné zadat vlastní začátek zpracovávaného signálu. Tím je možno opravit případy, kdy v okamžiku vypuštění není maximum signálu.

# Která data jsou označena jako nevalidní?

* Pokud má signál záznam v souboru `csv/FFT_failed.csv`, označí se jako špatný a nezpracovává
  se do výsledných statistik. Generuje se průběh signálu a výstup FFT zkontrolovat po stažení.
  Stahování je na volbě Download v modrém pásu.
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


def Welch_interactive():
    with solara.Row():
        solara.Markdown(r"$n$ (where $\text{nperseg}=2^n$)")
        solara.ToggleButtonsSingle(values=list(range(6, 13)), value=n)
    data = zpracuj(type='welch')
    df_fft = data['welch']  # .loc[:restrict]
    ymax = df_fft.to_numpy().max()
    if probe.value in ["Pt3", "Pt4"]:
        df_fft.columns = [probe.value]
    figFFT = px.line(df_fft,
                     height=s.height.value, width=s.width.value,
                     title=f"Welch spectrum: {s.method.value}, {s.day.value}, {s.tree.value}, {s.measurement.value}, {probe.value}",
                     log_y=True,  # range_y=[ymax/1000000, ymax*2]
                     )
    figFFT.update_layout(xaxis_title="Freq/Hz", yaxis_title="FFT amplitude")
    solara.FigurePlotly(figFFT, on_click=save_freq_on_click)
            

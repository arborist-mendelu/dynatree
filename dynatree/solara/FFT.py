#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:24:26 2024

@author: marik
"""

import plotly.express as px
import solara
import dynatree.solara.select_source as s
from dynatree import dynatree, plot_spectra_for_probe
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal
from solara.lab import task
import matplotlib.pyplot as plt
import os
import logging
from io import BytesIO
from FFT_spectrum import extend_dataframe_with_zeros
# from lib_dynasignal import do_fft, process_signal

logger = logging.getLogger("Solara_FFT")
logger.setLevel(logging.INFO)

filelogger = logging.getLogger("FFT Rotating Log")
filelogger.setLevel(logging.INFO)
filehandler = logging.handlers.RotatingFileHandler(f"{os.path.expanduser('~')}/solara_log/solara_FFT.log", maxBytes=10000000, backupCount=10)
log_format = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s")
filehandler.setFormatter(log_format)
filelogger.addHandler(filehandler)


DT = 0.01
pd.set_option('display.max_rows', 5000)

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
    save_button_color.value = "red"

def plot():
    data_obj = dynatree.DynatreeMeasurement(
        day=s.day.value, tree=s.tree.value, measurement=s.measurement.value, measurement_type=s.method.value)
    if len(probe_inclino.value)>0:
        logger.debug("Using Dataframe for pulling")
        df = data_obj.data_pulling
        df = df[probe.value]
    elif len(probe_optics.value)>0:
        logger.debug("Using Dataframe with optics")
        df = data_obj.data_optics
        mask = [i for i in df.columns if "Y0" in i[1]]
        df = df.loc[:,mask]
        df.columns = [i[0] for i in df.columns]
        df = df - df.iloc[0,:]
        df = df[probe.value]
    else: 
        logger.debug("Using Dataframe for ACC")
        df = data_obj.data_acc        
        df = df[probe.value]

    fig = px.scatter(df, height = s.height.value, width=s.width.value,
             title=f"Dataset: {s.method.value}, {s.day.value}, {s.tree.value}, {s.measurement.value}",  
             **kwds)
    solara.FigurePlotly(fig, on_click=set_click_data)

    return df

kwds = {"template": "plotly_white", 
        }

probe = solara.reactive([])
probe_inclino = solara.reactive([])
probe_optics = solara.reactive([])
probe_acc = solara.reactive([])
choices_disabled = solara.reactive(False)
t_from = solara.reactive(0)
t_to = solara.reactive(0)
remark = solara.reactive("")
# peaks = solara.reactive("")
fft_freq = solara.reactive("")
save_button_color = solara.reactive("none")
window_function = solara.reactive("none")
window_functions = ["none", "zeros", "hanning", "tukey"]
alpha = solara.reactive(0.1)
# acc_fft_axis = solara.reactive('a03_z')

tab_index = solara.reactive(0)

@task
def prepare_images_for_comparison(): 
    logger.debug(f"prepare_images_for_comparison started, experiment_changed.value is {experiment_changed.value}")
    ans = lib_plot_spectra_for_probe.plot_spectra_for_all_probes(
        measurement_type=s.method.value, 
        day=s.day.value, 
        tree=s.tree.value,
        measurement=s.measurement.value, 
        fft_results=df_limits.value, 
        log_x = log_x.value,
        xmax = range_x.value,
        )
    plt.close('all')
    logger.debug("prepare_images_for_comparison finished")
    return ans

# filetype = solara.reactive("png")

def SrovnaniReset():
    experiment_changed.value = False
    prepare_images_for_comparison()


@solara.component
def Srovnani():
    logger.debug(f"******** Srovnani entered, resetuj is {resetuj}, experiment_changed.value is {experiment_changed.value}")
    if prepare_images_for_comparison.not_called or experiment_changed.value:
        logger.info(f"Comparison available on buton click. experiment_changed.value is {experiment_changed.value}")
        solara.Info("The computation for this tab has not been started yet or it does not fit the current data. (The tree or date or something like this has been changed.) Run the computation manualy by clicking the Start button.")
        with solara.Row():
            solara.Button(label="Start", on_click=SrovnaniReset)
        return
    # with solara.Row():
        # solara.ToggleButtonsSingle(value=filetype, values=["png","svg"], mandatory=True)
    if not prepare_images_for_comparison.finished:
        with solara.Row():
            solara.Text("Pracuji jako ďábel. Může to ale nějakou dobu trvat.")
            solara.SpinnerSolara(size="100px")
            return    
    ans = prepare_images_for_comparison.value
    if ans == None:
        solara.Error("Mhm, data not available")
    else:
        with solara.Row():
            solara.Markdown(f"**Experiment {s.method.value}, {s.day.value}, {s.tree.value},  {s.measurement.value}**")
            solara.Button(label="Redraw", on_click=prepare_images_for_comparison)
            solara.Text("(Use if the images do not match the selection on the left panel. The need for this should appear only of you change x-axis setting.)")
        with solara.ColumnsResponsive([6,6], large=[4,4,4], xlarge=[4,4,4]):
            for i in ans:
                with solara.Card(title=i, style={'background-color':"#FFF"}):
                    with solara.VBox():
                        solara.FigureMatplotlib(ans[i]['fig'], 
                                                # format=filetype.value
                                                )
                        solara.Text(f"Peaks:{ans[i]['peaks']}")
                        solara.Text(ans[i]['remark'])
    plt.close('all')
    
def resetuj(x=None):
    # Srovnani(resetuj=True)
    s.measurement.set(s.measurements.value[0])
    generuj_obrazky()
    
# The following is true if day or tree or something changes outside tab 1.
# In this acase we should remove the images created there.    
experiment_changed = solara.reactive(False)
    
def generuj_obrazky(x=None):
    if tab_index.value == 1:
        prepare_images_for_comparison()
    elif tab_index.value == 0:
        experiment_changed.value=True
        DoFFT()
        logger.debug("Funkce RESETUJ")
    else:
        experiment_changed.value=True

range_x = solara.reactive(3)
log_x = solara.reactive(False)

@solara.component
def Page():
    solara.Title("DYNATREE: FFT")
    solara.Style(s.styles_css)
    with solara.Sidebar():
        s.Selection(exclude_M01=True, 
                    optics_switch=False,       
                    day_action = resetuj,
                    tree_action = resetuj,
                    measurement_action = generuj_obrazky)
        with solara.Card(title="FFT parameters"):
            with solara.Column():
                solara.Switch(label="log freq axis", value=log_x)
                with solara.Tooltip("Upper bound for image with FFT spectrum."):
                    with solara.Column():
                        solara.SliderInt(label="freq upper boundⓘ", value=range_x, min=3, max=20)
                with solara.Tooltip("Implementační průběžná pokusná fáze. Okno se bere v úvahu při tvorbě FFT na této stránce, ale informace o použitém okně se zatím neukládá společně s hodnotami peaků a ani se nastavení nezohledňuje na stránce se přehledem všech zpracovaných probů pro dané měření."):
                    with solara.Column():
                        solara.SliderValue(label="windowⓘ", value=window_function, values=window_functions)
                        solara.SliderFloat(label="tukey alpha", value=alpha, min=0, max=1, step=0.01)
                with solara.Tooltip("This button redraws images on the 'Srovnani' tab. Use after a change in the x-axis setting."):
                    with solara.Column(align='center'):
                        solara.Button(label="Redraw", on_click=prepare_images_for_comparison)
        s.ImageSizes()

    if s.measurement.value not in s.available_measurements(s.df.value, s.day.value, s.tree.value, s.method.value, exclude_M01=True):
        print(f"Mereni {s.measurement.value} neni k dispozici, koncim")
        return
    preload_data()
    solara.Warning("Tento postup asi odnesl čas, tady asi být nechceš.")
    with solara.lab.Tabs(value=tab_index):
        with solara.lab.Tab("FFT"):
            if tab_index.value == 0:
                try:
                    DoFFT()
                except:
                    with solara.Error():
                        solara.Markdown(
    """
    **Bohužel nastala nějaká chyba.**
    
    * Zkus nahlásit při jaké činnosti a při jaké volbě měření a sledovaných veličin. 
    * Možná jsou špatné meze. Je určitě dolní mez menší než horní?
    """)
        with solara.lab.Tab("Srovnání"):
            with solara.Tooltip(                
                solara.Markdown(
                    """
                    
                    * Tady by měly být pro vybraný experiment zpracovaná FFT, tj. ta, kde je ručně potvrzen aspoň jeden peak, nebo je vypsána poznámka.
                    * Exntenzometr a inkinometry maji nakopirovane stejne zacatky a konce. Pokud to nekdo 
                    rucne nezmenil, tak to muze vest k tomu, ze je presna shoda v peacich.
                    
                    """,style={'color':'white'})):
                    with solara.Column():
                        solara.Markdown("**Srovnání probůⓘ.**")
            try:
                logger.debug("Zalozka Srovnani")  
                # If another tab is selected, do not run the computation, but delete the old one here.
                if tab_index.value == 1:
                    Srovnani()                    
            except:
                solara.Error("Něco se nepovedlo. Možná není žádné meření zpracované")

        with solara.lab.Tab("Návod&Download"):
            Navod()

def ChooseProbe():
    with solara.Tooltip(solara.Markdown(
            """
            * The probes are divided to three groups according to the device. The topmost choice is active. 
            * In particular, to see optics probes, nether Elasto nor Inclino should be checked. To see ACC, neither optics probe nor Elasto or Inclino should be checked.
            * The most interesting axis for acc is z axis.
            """, style={'color':'white'})):
        with solara.Column():
            solara.Markdown("**Probesⓘ**")
    data_obj = dynatree.DynatreeMeasurement(
        day=s.day.value, tree=s.tree.value, measurement=s.measurement.value, measurement_type=s.method.value)
    probes_inclino = ["Elasto(90)","Inclino(80)X","Inclino(80)Y","Inclino(81)X","Inclino(81)Y"]
    probes_optics = ["Pt3","Pt4"] + [f"BL{i}" for i in range(44,68)]
    probes_acc = ['a01_x', 'a01_y', 'a01_z', 'a02_x', 'a02_y', 'a02_z', 
                  'a03_x', 'a03_y', 'a03_z', 'a04_x', 'a04_y', 'a04_z']
    probes =  probes_inclino + probes_optics + probes_acc 
    with solara.Column():
        solara.ToggleButtonsMultiple(value=probe_inclino, values=probes_inclino)
        solara.ToggleButtonsMultiple(value=probe_optics, values=probes_optics)
        solara.ToggleButtonsMultiple(value=probe_acc, values=probes_acc)
        if len(probe_inclino.value) != 0:
            probe.value = probe_inclino.value
        elif len(probe_optics.value) != 0:
            probe.value = probe_optics.value
        elif len(probe_acc.value) != 0:
            probe.value = probe_acc.value
        else:
            solara.Info("Vyber probe. Stránka se bude automaticky aktualizovat při výběru probu nebo změně stromu, dne nebo měření.")
            return None
        solara.Info(f"Active probes: {probe.value}")
    # solara.ToggleButtonsMultiple(value=probe, values=probes, mandatory=True)
    if not data_obj.is_optics_available:
        solara.Warning(f"Optika není k dispozici pro {s.method.value} {s.day.value} {s.tree.value} {s.measurement.value}.")
    return data_obj

df_limits = solara.reactive(pd.read_csv("csv/solara_FFT.csv", index_col=[0,1,2,3,4], dtype={'probe':str}).fillna(""))
df_limits.value = df_limits.value.sort_index()

def reload_csv():
    df_limits.value = pd.read_csv("csv/solara_FFT.csv", index_col=[0,1,2,3,4], dtype={'probe':str}).fillna("")
    df_limits.value = df_limits.value.sort_index()
    
def on_file(f):
    try:
        df_limits.value = pd.read_csv(BytesIO(f['data']), index_col=[0,1,2,3,4], dtype={'probe':str}).fillna("")
        df_limits.value = df_limits.value.sort_index()
    except:
        solara.Error("Load failed. Reloading server data.")
        reload_csv()

    
def preload_data():
    logger.debug("preload data started")
    logger.debug(f"looking for {s.method.value} {s.day.value} {s.tree.value} {s.measurement.value} {probe.value}")
    if len(probe_inclino.value) != 0:
        probe.value = probe_inclino.value
    elif len(probe_optics.value) != 0:
        probe.value = probe_optics.value
    elif len(probe_acc.value) != 0:
        probe.value = probe_acc.value    
    if len(probe.value) == 0:
        logger.debug("Nothing selected, finish")
        return None
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
    logger.debug("preload data finished")


@solara.component
def FFT_parameters():
    logger.debug("Inside FFT prameters")
    with solara.Row():
        with solara.Tooltip("Hodnoty je možné zadat číslem do políčka nebo kliknutím na bod v grafu výše. Se shiftem se nastavuje konec časového intervalu."):
            with solara.Column():
                solara.Markdown("**Limits for FFTⓘ:**")
        solara.InputFloat("From",value=t_from)
        solara.InputFloat("To",value=t_to)
        SaveButton()
    if pd.isna(t_to.value):
        t_to.value = 0
    if pd.isna(t_from.value):
        t_from.value = 0

@solara.component
def SaveButton():
    solara.Button(label="Save to table", on_click=save_limits, color=save_button_color.value)
    
def save_limits():
    if len(probe.value) == 0:
        solara.lab.ConfirmationDialog(True, content="Select at least one variable.")
        return
    df_limits.value.loc[(
        s.method.value, s.day.value, s.tree.value, s.measurement.value, probe.value[0]),:
        ] = [t_from.value, t_to.value, fft_freq.value, remark.value]
    df_limits.value = df_limits.value.sort_index()
    save_button_color.value = "none"
    filelogger.info(f"Saved {s.method.value},{s.day.value},{s.tree.value},{s.measurement.value},{ probe.value[0]},{t_from.value},{t_to.value},{fft_freq.value},{remark.value}")

@solara.component
def DoFFT():
    logger.debug(f"DoFFT entered {s.day.value} {s.tree.value} {s.measurement.value}" )
    with solara.Card():
        data_obj = ChooseProbe()
    if data_obj is None:
        # stop if no probe is selected
        return None
    with solara.ColumnsResponsive(xlarge=[6,6]):
        with solara.Column():
            with solara.Card():
                try:
                    df = plot()
                except:
                    solara.Warning("Nepovedlo se nakreslit graf. Možná nejsou zdrojová data?")
                    return

            with solara.Card():
                logger.debug("Will call FFT parameteres")
                FFT_parameters()
                # breakpoint()
                logger.debug("After call of FFT parameteres")
                if (t_to.value == 0) or (t_to.value < t_from.value): 
                    subdf = df.interpolate(method='index').loc[t_from.value:,:]
                else:
                    t_final = t_to.value
                    subdf = df.interpolate(method='index').loc[t_from.value:t_final,:]
                
                # Find new dataframe, resampled and restricted
                oldindex = subdf.index
                if len(oldindex) < 1:
                    logger.debug("Empty oldindex")
                    with solara.Warning():
                        solara.Markdown("Something unusual happened. The limits are probably not related to the signal.")
                    return
                # breakpoint()
                newindex = np.arange(oldindex[0],oldindex[-1], DT)
                newdf = pd.DataFrame(index=newindex, columns=subdf.columns)
                for i in subdf.columns:
                    newdf[i] = np.interp(newindex, oldindex, subdf[i].values)
                # subtract mean from each column
                newdf = newdf.apply(lambda col: col - col.mean())
                # apply window function
                if window_function.value == "zeros":
                    newdf = extend_dataframe_with_zeros(newdf, tail=5)
                elif window_function.value == "hanning":
                    hanning_window = np.hanning(len(newdf))
                    newdf = newdf.apply(lambda col: col * hanning_window)
                elif window_function.value == "tukey":
                    tukey_window = signal.windows.tukey(len(newdf), alpha=alpha.value, sym=False)
                    newdf = newdf.apply(lambda col: col * tukey_window)
                # solara.display(newdf.head())
                logger.debug("Will plot the detail.")
                try:
                    fig = px.scatter(newdf, height = s.height.value, width=s.width.value,
                                          title=f"Dataset: {s.method.value}, {s.day.value}, {s.tree.value}, {s.measurement.value}<br>Detail from {newdf.index[0]:.2f} to {newdf.index[-1]:.2f} resampled with dt={DT}", 
                                          **kwds)
                    solara.FigurePlotly(fig, on_click=set_click_data)    
                except:
                    solara.Error("Něco se nepovedlo. Možná meze nedávají smysl. Prověř meze.")
                    return

        with solara.Column():    
            with solara.Card():
                ShowFFTdata()
                    
                # get FFT output
                time_fft = newdf.index.values    
                N = time_fft.shape[0]  # get the number of points
                xf_r = fftfreq(N, DT)[:N//2]
                df_fft = pd.DataFrame(index=xf_r, columns=newdf.columns)
                upper_b=1
                for col in newdf.columns:
                    signal_fft = newdf[col].values
                    time_fft = time_fft - time_fft[0]
                    signal_fft = signal_fft - np.nanmean(signal_fft) # mean value to zero
                    yf = fft(signal_fft)  # preform FFT analysis
                    yf_r = 2.0/N * np.abs(yf[0:N//2])
                    df_fft[col] = yf_r
                    upper_b = max(upper_b,10**(np.trunc(np.log10(np.max(yf_r)))+1))
            
                lower_b = upper_b/100000
                if log_x.value:
                    xmin = df_fft.index[1]
                else:
                    xmin = 0
                figFFT = px.line(df_fft, height = s.height.value, width=s.width.value,
                                      title=f"FFT spectrum for {s.method.value}, {s.day.value}, {s.tree.value}, {s.measurement.value}<br>Limits: from {newdf.index[0]:.2f} to {newdf.index[-1]:.2f}", 
                                      log_x=log_x.value, range_x=[xmin,range_x.value], 
                                      log_y=True, range_y=[lower_b,upper_b],  
                                      **kwds)
                figFFT.update_layout(xaxis_title="Freq/Hz", yaxis_title="FFT amplitude")
                solara.FigurePlotly(figFFT, on_click=save_freq_on_click)    
            
            ShowSavedData()


@solara.component
def ShowFFTdata():
    logger.debug(f"ShowFFTdata entered {fft_freq.value}")
    with solara.Row():
        solara.Markdown(f"**FFT freq**: {fft_freq.value}")
        solara.Button(label="Erase", on_click=smazat_fft)
        SaveButton()
        solara.Text("The first variable (blue) is considered as a label when saved.")
    solara.InputText("Remark", value=remark)


def smazat_fft(x=None):
    fft_freq.value = ""
    save_button_color.value = "red"
    
# filter_method = solara.reactive(False)
filter_day = solara.reactive(True)
filter_tree = solara.reactive(True)
filter_probe = solara.reactive(False)
choice_elasto = solara.reactive(False)
choice_pt3 = solara.reactive(False)
# filtered_df = solara.reactive(pd.DataFrame())

@solara.component
def ShowSavedData():
    # show saved data    
    with solara.Card():
        with solara.Tooltip(solara.Markdown(
                """
                * The data are from csv/solara_fft.csv.
                * If you change the data, download the file and someone has to merge the data
                  with the file on server.
                * If you filter out a02 (accelerometer near extensomter), you may want to see Elasto(90) as well. Similarly for a03 and Pt3. Just switch the corresponding switch on.
                """, style={'color':'white'})):
            with solara.Column():
                solara.Markdown("**Table with dataⓘ**")
        with solara.Card():
            solara.Markdown("**Data restrictions** (The data table is long and you probably do not want to see all the data here.)")
            with solara.Row():
                solara.Switch(label=f"Day {s.day.value}", value=filter_day)    
                solara.Switch(label=f"Tree {s.tree.value}", value=filter_tree)    
                solara.Switch(label=f"Probe {probe.value[0]}", value=filter_probe)    
                solara.Switch(label=" & Elasto", value=choice_elasto)    
                solara.Switch(label=" & Pt3", value=choice_pt3)    
        logger.debug(f"ShowSavedData entered {filter_day.value} {filter_tree.value} {filter_probe.value}")
        try:
            filtered_df = df_limits.value.copy()
            if filter_day.value:
                filtered_df = filtered_df.loc[
                    (slice(None), s.day.value,slice(None),slice(None),slice(None)), :]
            if filter_tree.value:
                filtered_df = filtered_df.loc[
                    (slice(None), slice(None), s.tree.value,slice(None),slice(None)), :]
            # Add elasto?
            if choice_elasto.value and probe.value[0] != "Elasto(90)":
                filtered_df_elasto = filtered_df.loc[
                    (slice(None), slice(None),slice(None),slice(None),"Elasto(90)"), :]
            else:
                filtered_df_elasto = pd.DataFrame()                
            # Add PT3?
            if choice_pt3.value and probe.value[0] != "Pt3":
                filtered_df_pt3 = filtered_df.loc[
                    (slice(None), slice(None),slice(None),slice(None),"Pt3"), :]
            else:
                filtered_df_pt3 = pd.DataFrame()                
            # Filter probe according to the choice and merge
            if filter_probe.value:
                filtered_df = filtered_df.loc[
                    (slice(None), slice(None),slice(None),slice(None),probe.value[0]), :]
            filtered_df = pd.concat([filtered_df, filtered_df_elasto, filtered_df_pt3]).sort_index()
            # df_limits.value = tempdf
            solara.display(filtered_df)
        except Exception as e:
            logger.error(f"ShowSavedData failed {e}")
        with solara.Row():
            solara.FileDownload(df_limits.value.to_csv(), filename=f"solara_FFT.csv", 
                                label=f"Download as csv")
            solara.Button(label="Drop all rows", on_click=drop_rows)
            solara.Button(label="Reload from server", on_click=reload_csv)
            solara.FileDrop(label="You may upload your own csv here. Drag the file to this area.", on_file = on_file, lazy=False)
        solara.Markdown(
        f"""
        rows in csv: {df_limits.value.shape[0]}
                      
        rows with peak or remark filled: {df_limits.value[(df_limits.value['peaks'].notna()) & (df_limits.value['peaks'] != '') | 
                        (df_limits.value['remark'].notna()) & (df_limits.value['remark'] != '')].shape[0]}              
        """
            )

def drop_rows():
    df_limits.value = df_limits.value.head(0)

@solara.component
def Navod():
    with solara.Card(style={'background-color':"#FBFBFB"}):
        solara.Markdown("**Downloads** Soubor ke stažení obsahuje zpracované proby, začátek a konec intervalu, ručně stanovené peaky, případnou poznámku")
        with solara.Row(justify="space-around",style={'background-color':"#FBFBFB"}):
            solara.FileDownload(df_limits.value.to_csv(), filename="fft_old_dynatree.csv", label="Peaks")
 
    solara.Markdown(
"""
**TL;DR**

* Klikáním na tlačítka vyber zdroj dat a která data chceš zkoumat. Uvažují se jenom veličiny z prvního 
  neprázdného menu.
* Je možné vybrat více voleb. 
* Volba Elasto eliminuje všecny ostatní případné volby.
* Nastav časový interval na kterém chceš dělat FFT zapsáním hodnot do políček nebo kliknutím na bod v grafu (koncový bod se shiftem).

**Postup**

* Měření vyber v levém panelu. V horním menu vyber sledované veličiny.
* Pokud vybereš "Elasto", použijí se data z extenzometru a další volby se ignorují. Jinak
  se zpracovává podle optiky vše zatržené. Pokud není nic zatržené z optiky, zpracovávají se data
  z akcelerometru. Pokud není zatrhnuté vůbec nic, zpracovává se extenzometr.
* Vše se interpoluje na 0.01s, to je v souladu se vzorkovaci frekvenci optiky.
* Podle grafu můžeš vybrat rozsah pro FFT. Začátek a konec se zapisuje do políček pod grafem nebo se dá naklikat v grafu. Klik=dolní mez, Shift+Klik=horní mez. 
  Klikat se dá do kteréhokoliv z obou grafů. I do celkového i do detailního.
* Nulová horní mez znamená rozsah až do konce. Raději ale dát explicitní konec, aby byla v tabulce zaznamenána délka intervalu. Ta něco také vypovídá o spolehlivosti frekvence.
* BL jsou konce probů typu BendLine. Všechny výchylky jsou brány v ose y a je uvažována změna oproti výchozímu stavu, tj. uvažujeme posunutí, ne absolutní souřadnice v prostoru.
* Uložit peak znamená uložit do paměti. Potom je potřeba ještě stáhnout csv s daty a případně podle něj aktualizovat uložená data na serveru. 
* Pokud něco spadne před stažením dat, každé uložení peaku se loguje a je možné ztracená data vytáhnout z logu `~/solara_log/solara_FFT.log` nebo tak nějak.

**Co je to za data**

* Data z optiky, tahovek, acc, interpolovana (aby se odstranily nan) a presamplovana na 0.01s.
* Intervaly pro optiku a komentare byly rucne stanoveny drive, ted pretazeno, nekdy upraveno, napr doplneny druhy peak.
* Inervaly pro elasto naklikany rucne, potom se stejna data prekopirovala na inklinometry a peaky pro elasto a inklino byly naklikany rucne.


**Poznámky**

* Každý přístroj se bere přímo z měření. Není proto ACC nebo exenozmetr 
  synchronizován s optikou. Pokud jsou k dispozici všechny údaje, mohou se 
  časy maličko lišit posunutím.
* Kontroloval jsem oproti datům od Patrika, který má 2021-03-22 zpracované v Matlabu. Data se trochu liší, ale to bude asi tím, kde přesně se odřezávalo. Jinak na
  stejném časovém intervalu počítá Matlab 5000Hz (příkazy z <https://www.mathworks.com/help/matlab/ref/fft.html>) stejně jako Python 100Hz.
"""
        )


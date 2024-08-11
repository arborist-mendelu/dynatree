import sys
import glob
import matplotlib.pyplot as plt
#sys.path.append('./ERC/ERC/Mereni_Babice_zpracovani/skripty/')
# from extract_release_data import find_release_data_one_measurement
import lib_dynatree as lt
import pandas as pd
import numpy as np

def split_path(file):
    data = file.split("/")
    data[-1] = data[-1].replace(".csv","")
    return [file,data[-2].replace("_","-")] + data[-1].split("_")

def get_all_measurements(cesta="../data"):
    """
    Get dataframe with all measurements. The dataframe has columns
    date, tree and measurement.
    """
    files = glob.glob(cesta+"/csv/*/BK*.csv")        
    out = [split_path(file) for file in files]
    df = pd.DataFrame([i[1:] for i in out], columns=['day','tree', 'measurement'])
    df = df.sort_values(by=list(df.columns))
    df = df.reset_index(drop=True)
    return df
def available_measurements(df, day, tree):
    select_rows = (df["day"]==day) & (df["tree"]==tree)
    values = df[select_rows]["measurement"].values
    return list(values)

df = get_all_measurements()
days = df["day"].drop_duplicates().values
trees = df["tree"].drop_duplicates().values
measurements = df["measurement"].drop_duplicates().values

import solara
import time
from solara.lab import task

SOLARA_PROXY_CACHE_DIR = "/tmp/solara2"
day = solara.reactive(days[0])
tree = solara.reactive(trees[0])
measurement = solara.reactive(measurements[0])

from lib_dynatree import read_data
from csv_add_inclino import extend_one_csv
from plot_probes_inclino_force import plot_one_measurement

def reset_measurement(a):
    measurement.set(measurements[0])

@task
def nakresli():
    if measurement.value not in available_measurements(df, day.value, tree.value):
        fig, ax = plt.subplots()
        return fig
    fig = plot_one_measurement(
            date=day.value,
            tree=tree.value, 
            measurement=measurement.value, 
            path="../data",
            # xlim=(0,50),
            # df_extra=df_ext,
            # df=DF
            return_figure=True
            )    
    return fig
    
@solara.component
def Page():
    solara.Title("Oscillation: optics, inclinometers, elastometer, force synchro")
    with solara.Sidebar():
        solara.Markdown(
            """
            ## Oscilace, inklinometry, elastometr

            Vyber si den, strom a měření a klikni na tlačítko pro spuštění výpočtu.
            """
            )
    with solara.Card():
        with solara.Column():
            solara.ToggleButtonsSingle(value=day, values=list(days))
            solara.ToggleButtonsSingle(value=tree, values=list(trees))
            solara.ToggleButtonsSingle(value=measurement, values=list(measurements))
        with solara.Row():
            solara.Info(f"Day {day.value}, Tree {tree.value}. Available measurements: {available_measurements(df, day.value, tree.value, )}")
            solara.Markdown(f"**Selected**: {day.value}, {tree.value}, {measurement.value}")
            solara.Button("Run calculation", on_click=nakresli, color="primary")
    solara.ProgressLinear(nakresli.pending)    
    if measurement.value not in available_measurements(df, day.value, tree.value):
        solara.Error(f"Measurement {measurement.value} not available for this tree.")
        return
    if nakresli.finished:
        plt.show(nakresli.value)
    elif nakresli.not_called:
        solara.Text("Vyber měření a stiskni tlačítko Run calculation")
    else:
        with solara.Row():
            solara.Text("Pracuji jako ďábel. Může to ale nějakou dobu trvat.")
            solara.SpinnerSolara(size="100px")

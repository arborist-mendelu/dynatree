#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:00:04 2024

@author: marik
"""

DATA_PATH = "../data"

navod = """
Návod se napíše později.
"""

title = "Pulling, force, inclinometers, elastometer"

import solara
from solara.lab import task
import glob
import numpy as np
import pandas as pd
import static_pull
import matplotlib.pyplot as plt


def split_path(file):
    data = file.split("/")
    data[-1] = data[-1].replace(".TXT","")
    return [file,data[-2].replace("_","-")] + data[-1].split("_")

def get_all_measurements(cesta=DATA_PATH):
    """
    Get dataframe with all measurements. The dataframe has columns
    date, tree and measurement.
    """
    files = glob.glob(cesta+"/pulling_tests/*/BK_??_M?.TXT")       
    files = [i.replace("BK_","BK") for i in files]
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

day = solara.reactive(days[0])
tree = solara.reactive(trees[0])
measurement = solara.reactive(measurements[0])

@task
def nakresli():
    return static_pull.nakresli(day.value, tree.value, measurement.value)
    
@solara.component
def Page():
    solara.Title(title)
    solara.Style(".widget-image{width:100%}")
    with solara.Sidebar():
        solara.Markdown(navod)
    with solara.Card():
        with solara.Column():
            solara.ToggleButtonsSingle(value=day, values=list(days), on_value=lambda x: measurement.set(measurements[0]))
            solara.ToggleButtonsSingle(value=tree, values=list(trees), on_value=lambda x: measurement.set(measurements[0]))
            with solara.Row():
                solara.ToggleButtonsSingle(value=measurement, 
                                           values=available_measurements(df, day.value, tree.value),
                                           on_value=lambda x:nakresli()
                                           )
        solara.Div(style={"margin-bottom": "10px"})
        with solara.Row():
            solara.Button("Run calculation", on_click=nakresli, color="primary")
            solara.Markdown(f"**Selected**: {day.value}, {tree.value}, {measurement.value}")    
    solara.ProgressLinear(nakresli.pending)    
    if measurement.value not in available_measurements(df, day.value, tree.value):
        solara.Error(f"Measurement {measurement.value} not available for this tree.")
        return
    
    if nakresli.finished:
        f = nakresli.value
        with solara.ColumnsResponsive(6): 
            for _ in f:
                solara.FigureMatplotlib(_)
    elif nakresli.not_called:
        solara.Text("Vyber měření a stiskni tlačítko Run calculation")
    else:
        with solara.Row():
            solara.Text("Pracuji jako ďábel. Může to ale nějakou dobu trvat.")
            solara.SpinnerSolara(size="100px")
            
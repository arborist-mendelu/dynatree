# -*- coding: utf-8 -*-
import solara
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
import dynatree.solara.select_source as s
import glob
import numpy as np

active_columns = solara.reactive([])
def reset_columns():
    active_columns.value = []

@solara.component
def Page():
    # display(df)
    # display(all_columns)

    solara.Style(s.styles_css)
    solara.Title("DYNATREE: Soil, transpiration, air condition, ...")

    with solara.lab.Tabs(lazy=True):
        with solara.lab.Tab("Trans vse"):
            df = read_csv_to_df()
            solara.ToggleButtonsMultiple(value=active_columns, values=list(df.columns))
            solara.Button(label="Reset", on_click=reset_columns)
            draw_graphs(df)
        with solara.lab.Tab("Penetrologger"):
            penetrologger()

def read_csv_to_df():
    df = pd.read_csv(config.file["trans_vse.csv"])
    df["Time"] = pd.to_datetime(df["Time"])  # , errors="coerce")
    df = df.drop(columns=["Unnamed: 0"])
    df = df.set_index("Time")
    return df

@solara.component
def draw_graphs(df):
    if len(active_columns.value) == 0:
        active_columns.value = [df.columns[0]]
    df_plot = df[active_columns.value]
    num_cols = len(df_plot.columns)

    fig = make_subplots(rows=num_cols, cols=1, shared_xaxes=True, subplot_titles=df_plot.columns,
                        vertical_spacing=0.1/num_cols)
    for i, col in enumerate(df_plot.columns, start=1):
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], mode="lines", name=col), row=i, col=1)
    fig.update_layout(height=300 * num_cols, title="", showlegend=False)

    solara.FigurePlotly(fig)

@solara.component
def penetrologger():
    def fixname(string):
        string = string.split('/')[-1]
        string = string.split('.')[0]
        string = string.replace("penetrologger ","")
        return string
    def fix_column_name(string):
        if not ("Unnamed" in string):
            return string
        string = string.replace("Unnamed: ", "")
        string = int(string)-4
        return string

    # Funkce, která nahradí nuly na konci řádku za NaN
    def replace_trailing_zeros(row):
        # Zjistí, kde jsou nuly na konci řádku a nahradí je NaN
        for i in range(len(row) - 1, -1, -1):  # Procházíme řádek zpětně
            if row.iloc[i] == 0:
                row.iloc[i] = np.nan  # Nahradíme nulu NaN
            else:
                break  # Jakmile narazíme na hodnotu jinou než nula, zastavíme
        return row

    files = glob.glob(config.file["penetrologgers"])
    data = {fixname(file): pd.read_excel(file) for file in files}
    # data = {file: data[file].insert(0,'file',file) for file in files}
    # for key in data.keys():
    #     data[key].insert(0,'file', fixname(key))
    # display(data)
    df = pd.concat(data)
    df = df.reset_index()
    df = df.drop( df.columns[1], axis=1)
    df.columns.values[0] = 'Day'
    df.columns = [fix_column_name(i) for i in df.columns]
    df = df.apply(replace_trailing_zeros, axis=1)
    # display(df["Day"].drop_duplicates())
    # return
    # df["Day"] = pd.to_datetime(df["Day"], format='%Y%m%d')

    # df = pd.read_excel(config.file["penetrologgers"])
    display(df)


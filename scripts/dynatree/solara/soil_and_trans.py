# -*- coding: utf-8 -*-
import solara
import pandas as pd
import plotly.graph_objects as go
from docutils.nodes import label
from plotly.subplots import make_subplots
import config
import dynatree.solara.select_source as s

df = pd.read_csv(config.file["trans_vse.csv"])
# Převedení sloupce Time na datetime
df["Time"] = pd.to_datetime(df["Time"])#, errors="coerce")

# Odstranění nepotřebného sloupce, pokud je jen indexem
df = df.drop(columns=["Unnamed: 0"])
df = df.set_index("Time")

active_columns = solara.reactive([df.columns[0]])
all_columns = list(df.columns)

def reset_columns():
    active_columns.set([all_columns[0]])

@solara.component
def Page():
    # display(df)
    # display(all_columns)

    solara.Style(s.styles_css)
    solara.Title("DYNATREE: Soil, transpiration, air condition, ...")

    solara.ToggleButtonsMultiple(value=active_columns, values=all_columns)
    solara.Button(label="Reset", on_click=reset_columns)
    draw_graphs()

@solara.component
def draw_graphs():
    if len(active_columns.value) == 0:
        active_columns.value = [all_columns[0]]
    df_plot = df[active_columns.value]
    # display(df_plot)
    num_cols = len(df_plot.columns)
    fig = make_subplots(rows=num_cols, cols=1, shared_xaxes=True, subplot_titles=df_plot.columns)
    for i, col in enumerate(df_plot.columns, start=1):
        fig.add_trace(go.Scatter(x=df_plot.index, y=df_plot[col], mode="lines", name=col), row=i, col=1)
    fig.update_layout(height=300 * num_cols, title="Všechny sloupce v subgrafech", showlegend=False)
    solara.FigurePlotly(fig)

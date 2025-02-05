# -*- coding: utf-8 -*-
import solara
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import config
import dynatree.solara.select_source as s
import glob
import numpy as np
import matplotlib.pyplot as plt

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
            s.Selection_trees_only()
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
    df = pd.read_csv(config.file["penetrologger.csv"])
    solara.Markdown(f"# Tree {s.tree.value}")
    df = df[df["tree"]==s.tree.value]

    df_grouped = df.drop(["poznámka", "PENETRATION DATA", "tree"], axis=1).groupby(["day", "směr"],
                                                                                   as_index=False).mean()
    newdf = df_grouped.set_index(["směr", "day"]).sort_index().T

    # highlight_dates = ["2021-06-29", "2022-08-16", '2024-09-02', '2024-09-02_mokro']
    # existing_columns = [col for col in newdf.columns if col[1] in highlight_dates]

    # Funkce pro stylování buněk
    def highlight_text(val):
        return "color: red;"

    styled_df = (
        newdf.style.background_gradient(axis=None)
        .map(lambda x: 'color: lightgray' if pd.isnull(x) else '')
        .map(lambda x: 'background: transparent' if pd.isnull(x) else '')
    )

    with solara.Info():
        solara.Markdown(f"""
    * Data pro jeden strom. Svisle sleduj hodnoty jako funkci hloubky, vodorovne sleduj jak se ve stejne hloubce 
      hodnoty meni v case.
    * U stromu 13 a 10 to nejak nehraje.
    """, style={'color': 'inherit'})
    display(styled_df)

    solara.FileDownload(df.to_csv(), filename=f"penetrologger_{s.tree.value}.csv")

    with solara.Info():
        solara.Markdown("""
        * Časový vývoj pro jednotlivá místa
        * Letní měsíce jsou tečkovaně
        """, style={'color': 'inherit'})
    # Definice letních měsíců
    letni_mesice = {"06", "07", "08", "09"}

    fig, axs = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

    for ax, smer in zip(axs, np.unique([col[0] for col in newdf.columns])):
        for day, data in newdf[smer].items():
            linestyle = ":" if any(mesic in day for mesic in letni_mesice) else "-"
            linewidth = 3 if any(mesic in day for mesic in letni_mesice) else 2
            data.plot(ax=ax, linestyle=linestyle, label=day, linewidth=linewidth)

        ax.set(title=f"{s.tree.value} {smer}")
        ax.legend()
        ax.grid()

    solara.FigureMatplotlib(fig)
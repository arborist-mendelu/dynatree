#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 02:14:14 2023

@author: marik
"""

from dash import Dash, html, dcc, callback, Output, Input, State, ctx
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from lib_dynatree import do_fft
import os

pio.templates.default = "plotly_white"

# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)

days = ['2022-04-05','2022-08-16','2021-03-22','2021-06-29']
days.sort()

trees = ["01", "04", "07", "08", "09", "10", "11", "12", "13", "14", "16", "21", "24"]
trees = ["BK"+i for i in trees]

measurements = ["M02", "M03", "M04", "M05", "M06"]

probes = ["Pt3","Pt4"]

app.layout = dbc.Container([
    dbc.Row([
    html.Div('Měření Babice', className="text-primary text-center fs-3")
]),
    dbc.Row([dbc.Col(i) for i in [   
    dcc.RadioItems(days, value=days[0], id='radio-selection-1', labelStyle={'margin-right': '20px'}),
    dcc.RadioItems(trees, value=trees[0], id='radio-selection-2', labelStyle={'margin-right': '20px'}),
    dcc.RadioItems(measurements, value=measurements[0], id='radio-selection-3', labelStyle={'margin-right': '20px'}),
    dcc.RadioItems(probes, value=probes[0], id='radio-selection-4', labelStyle={'margin-right': '20px'}),
    ]]),    
    dbc.Row(
        [html.Span(id='result'),
         ]),
    dbc.Row([   
                dcc.Loading(
                id="ls-loading-1",
                children=[dcc.Graph(id='graph-content')],
                type="circle",
            )
    ]),
    # html.Button('Save', id='button', n_clicks=0),
    # html.P(id='placeholder'),
    dbc.Row([dbc.Col(html.P(id='start')),
    dbc.Col(html.P(id='end'))]),
    dbc.Row([   
                dcc.Loading(
                id="ls-loading-2",
                children=[dcc.Graph(id='graph-content-2')],
                type="circle",
            )
    ]),
], fluid=True)


# Podle radioboxu se načtou data, vykreslí se časový vývoj a uloží se sutovaný sloupec 
# do menší databáze.
@callback(
    Output('result', 'children'),
    Output('graph-content', 'figure'),
    *[Input(f'radio-selection-{i}', 'value') for i in [1,2,3,4]]
)
def update(date, tree, measurement, probe):
    _ = date.split("-")
    _.reverse()
    date = "".join(_)
    file = f"../01_Mereni_Babice_{date}_optika_zpracovani/csv/{tree}_{measurement}.csv"
    try:
        df = pd.read_csv(file,
                 header=[0,1], index_col=0, dtype = np.float64)    
    except:
        return f"Soubor {file} neexistuje",{}
    df.index=df["Time"].values.reshape(-1)
    data = df[(probe,"Y0")]
    data.columns = ["data"]
    data.to_csv("temp/data.csv")
    data = data.values
    data = data - data[0]
    time = df["Time"].values.reshape(-1)
    
    labels = {'x': "Time", 'y': "Position"}
    fig = px.line(x=time, y=data, labels=labels, title=f"{probe} movement")
    return f"file {file}",fig

# Pokud uživatel vybral část grafu, upraví se počáteční a koncová hodnota
# času pro FFT
@callback(
    Output('start', 'children'),
    Output('end', 'children'),
    Input('graph-content', 'relayoutData'),
    )
def update_fft_bounds(graph):
    df = pd.read_csv("temp/data.csv",
                  header=[0,1], index_col=0, dtype = np.float64)    
    if graph is not None and "xaxis.range[0]" in graph.keys():
        start = graph["xaxis.range[0]"]
    else:
        raise PreventUpdate
    if graph is not None and "xaxis.range[1]" in graph.keys():
        end = graph["xaxis.range[1]"]
    else:
        raise PreventUpdate
    return start, end


# FFT
@callback(
    Output('graph-content-2', 'figure'),
    Input('graph-content', 'relayoutData'),
    Input('start', 'children'),
    Input('end', 'children'),
    )
def update_fft(graph, start, end):
    if start is None:
        return {}
    df = pd.read_csv("temp/data.csv",
                  header=[0,1], index_col=0, dtype = np.float64)    
    limits = f"FFT from {start:.2f} to {end:.2f}"
    df = df[start:end]
    data=df.iloc[:,0].values
    time=df.index
    xf,yf = do_fft(data,time)
    fig2 = px.line(x=xf, y=yf, 
                    labels={'x': "Frekvence", 'y': 'Amplituda'}, 
                    log_y=True, range_x=[0,5],
                    title=limits)
    return fig2

# @callback(
#     Output('placeholder', 'children'),
#     *[Input(f'radio-selection-{i}', 'value') for i in [1,2,3,4]],
#     Input('start', 'children'),
#     Input('end', 'children'),
#     Input('button', 'n_clicks'),
#     prevent_initial_call = True
#     )
# def update_output(date, tree, measurement, probe, start, end, btn):
#     if "button" == ctx.triggered_id:
#         return f'From {start} to {end}.'
#     else:
#         raise PreventUpdate

    
if __name__ == '__main__':
    adresar = "temp"
    if not os.path.exists(adresar):
        os.makedirs(adresar)    
    app.run(debug=True)
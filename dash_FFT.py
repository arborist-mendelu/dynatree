#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 02:14:14 2023

@author: marik
"""

from dash import Dash, html, callback, Output, Input, State
from dash.exceptions import PreventUpdate
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from lib_dynatree import do_fft, do_welch
import lib_analyze_filenames as laf
import os
from lib_dash import csv_selection, make_graph

pio.templates.default = "plotly_white"

# Initialize the app - incorporate a Dash Bootstrap theme
# external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

probes = ["Pt3","Pt4"]

DF = pd.DataFrame()

app.layout =  dbc.Container(
    [
    html.H1('Měření Babice, FFT analýza', className="text-primary text-center fs-3"),
    *csv_selection(probes),
    make_graph(1),
    html.P(["Start time: ", html.Span(id='start'), " End time: ", html.Span(id='end')]),
    make_graph(2),
    ])

def vymaz_docasny_csv():
    global DF
    DF = pd.DataFrame()
    return None

# Zmena datumu nastavuje stromy, maze vyber mereni
@callback(
    Output('radio-selection-2', 'options', allow_duplicate=True),
    Output('radio-selection-2', 'value', allow_duplicate=True),
    Output('radio-selection-3', 'options', allow_duplicate=True),
    Output('radio-selection-3', 'value', allow_duplicate=True),
    Output('csv', 'children', allow_duplicate=True),
    Input('radio-selection-1', 'value'),
    prevent_initial_call=True
    )
def nastav_stromy(datum):
    vymaz_docasny_csv()
    return laf.days2trees[datum],None,[],None,None


# Zmena stromu nastavuje menu pro vyber mereni
@callback(
    Output('radio-selection-3', 'options', allow_duplicate=True),
    Output('radio-selection-3', 'value', allow_duplicate=True),
    Output('csv', 'children', allow_duplicate=True),
    Input('radio-selection-2', 'value'),
    State('radio-selection-1', 'value'),
    prevent_initial_call=True
    )
def nastav_mereni(strom,datum):
    vymaz_docasny_csv()
    if strom is None:
        raise PreventUpdate()
    return laf.day_tree2measurements[f"{datum}_{strom}"],None, None

# Zmena mereni ulozi jmeno souboru
@callback(
    Output('csv', 'children', allow_duplicate=True),
    Input('radio-selection-3', 'value'),
    *[State(f'radio-selection-{i}', 'value') for i in [1,2,4]],
    prevent_initial_call=True
    )
def sestav_csv(measurement, date, tree, probe):
    if measurement is None or date is None or tree is None:
        return None
    _ = date.split("-")
    _ .reverse()
    _ = "".join(_)
    file = f"../01_Mereni_Babice_{_}_optika_zpracovani/csv/{tree}_{measurement}.csv"
    return file

FILENAME = ""  # the last csv file which has been loaded and drawn

@callback(
    Output('graph-content-1', 'figure', allow_duplicate=True),
    Output('graph-content-2', 'figure', allow_duplicate=True),
    Input('csv', 'children'),
    Input('radio-selection-4', 'value'),
    *[State(f'radio-selection-{i}', 'value') for i in [1,2,3]],
    prevent_initial_call=True
    )    
def plot_graph(file,probe, date, tree, measurement):
    global DF
    global FILENAME
    if file is None:
        DF = pd.DataFrame()
        return {},{}
    if file != FILENAME:
        DF = pd.read_csv(file, header=[0,1], index_col=1, dtype = np.float64)
        DF = DF[[(i,"Y0") for i in probes]]
        DF = DF - DF.iloc[0,:]
        FILENAME = file
    fig = go.Figure(layout_title_text=f"{date} {tree} {measurement}: {probe}")
    fig.add_trace(go.Scatter(x=DF.index.values, y=DF[probe].values.reshape(-1), mode='lines', name=probe))
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Delta position")
    return fig,{}    


# Pokud uživatel vybral část grafu, upraví se počáteční a koncová hodnota
# času pro FFT
@callback(
    Output('start', 'children'),
    Output('end', 'children'),
    Output('message', 'children', allow_duplicate=True),
    Input('graph-content-1', 'relayoutData'),
    prevent_initial_call=True
    )
def update_fft_bounds(graph):
    if graph is not None and "xaxis.range[0]" in graph.keys():
        start = graph["xaxis.range[0]"]
    else:
        raise PreventUpdate
    if graph is not None and "xaxis.range[1]" in graph.keys():
        end = graph["xaxis.range[1]"]
    else:
        raise PreventUpdate
    return start, end, f"Limits updated to {start} and {end}."


# FFT
@callback(
    Output('graph-content-2', 'figure', allow_duplicate=True),
    Input('start', 'children'),
    Input('end', 'children'),
    State('radio-selection-4', 'value'),
    prevent_initial_call=True    
    )
def update_fft(start, end,probe):
    global DF
    if probe is None:
        raise PreventUpdate()
    if start is None:
        return {}
    df = DF[start:end].dropna()
    data=df.loc[:,probe].values.reshape(-1)
    time=df.index
 
    xf,yf = do_fft(data,time)
    f,Pxx = do_welch(data, time)

    fig = make_subplots(rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.02)
    
    fig.add_trace(go.Scatter(x=xf, y=yf, mode='lines+markers', name='FFT'
                    ), row=1, col=1)
    fig.add_trace(go.Scatter(x=f, y=Pxx, 
                             mode='lines+markers', name='Welch'
                    ),row=2, col=1)
    fig.update_yaxes(type="log")
    fig.update_layout(xaxis_range=[0,5])

    return fig


    
if __name__ == '__main__':
    adresar = "temp"
    if not os.path.exists(adresar):
        os.makedirs(adresar)    
    # app.run(debug=True)
    app.run()
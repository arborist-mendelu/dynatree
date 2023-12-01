#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 02:14:14 2023

@author: marik
"""

from dash import Dash, html, dcc, callback, Output, Input, State, callback_context
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from lib_dynatree import do_fft
import lib_analyze_filenames as laf
import os

pio.templates.default = "plotly_white"

# Initialize the app - incorporate a Dash Bootstrap theme
# external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

probes = ["Pt3","Pt4"]

app.layout =  dbc.Container(
    [
    html.H1('Měření Babice', className="text-primary text-center fs-3"),
    dbc.Row([
    *[dbc.Col(i) for i in [
    ["Day",dcc.RadioItems(laf.days, id='radio-selection-1', labelStyle={'margin-right': '20px'}, inline=True)],
    ["Tree",dcc.RadioItems([], id='radio-selection-2', labelStyle={'margin-right': '20px'}, inline=True)],
    ["Measurement",dcc.RadioItems([], id='radio-selection-3', labelStyle={'margin-right': '20px'}, inline=True)],
    ["Probe",dcc.RadioItems(probes, id='radio-selection-4', labelStyle={'margin-right': '20px'}, inline=True)],
    ]]]),
    dbc.Button('Load CSV', id='load-button'),html.Span(" "),
    dbc.Button('Plot', id='plot-button'),html.Span(" "),
    html.P(id='result', style={'color': 'gray'}),
    dcc.Loading(
                id="ls-loading-1",
                children=[dcc.Graph(id='graph-content')],
                type="circle",
            ),
    html.P(["Start: ", html.Span(id='start'), " End: ", html.Span(id='end')]),
    dcc.Loading(
                id="ls-loading-2",
                children=[dcc.Graph(id='graph-content-2')],
                type="circle",
            )])


TEMP_CSV_FILE = "temp/temp_data.csv"

def vymaz_docasny_csv():
    global TEMP_CSV_FILE
    try:
        os.remove(TEMP_CSV_FILE)
    except:
        pass
    return None


# Zmena datumu nastavuje stromy, maze vyber mereni a vybrany probe
@callback(
    Output('radio-selection-2', 'options', allow_duplicate=True),
    Output('radio-selection-3', 'options', allow_duplicate=True),
    Output('radio-selection-4', 'value', allow_duplicate=True),
    Output('graph-content', 'figure', allow_duplicate=True),
    Input('radio-selection-1', 'value'),
    prevent_initial_call=True
    )
def nastav_stromy(datum):
    vymaz_docasny_csv()
    return laf.days2trees[datum],[],None,{}


# Zmena stromu maze obrazek a vyber probu, nastavuje mereni
@callback(
    Output('radio-selection-3', 'options', allow_duplicate=True),
    Output('radio-selection-4', 'value', allow_duplicate=True),
    Output('graph-content', 'figure', allow_duplicate=True),
    Input('radio-selection-2', 'value'),
    State('radio-selection-1', 'value'),
    prevent_initial_call=True
    )
def nastav_mereni(strom,datum):
    vymaz_docasny_csv()
    return laf.day_tree2measurements[f"{datum}_{strom}"],None,{}

# Zmena mereni maze obrazek, docasny csv a vyber probu
@callback(
    Output('graph-content', 'figure', allow_duplicate=True),
    Output('radio-selection-4', 'value', allow_duplicate=True),
    Input('radio-selection-3', 'value'),
    prevent_initial_call=True
    )
def vymaz_graf(mereni):
    vymaz_docasny_csv()
    return {},None

# Podle radioboxu se načtou data a uloží se studované sloupce 
# do menší databáze. Maze se obrazek
@callback(
    Output('result', 'children',  allow_duplicate=True),
    Output('graph-content', 'figure', allow_duplicate=True),
    Output('graph-content-2', 'figure', allow_duplicate=True),
    Output('start', 'children', allow_duplicate=True),
    Output('end', 'children', allow_duplicate=True),
    Input('load-button', 'n_clicks'),
    *[State(f'radio-selection-{i}', 'value') for i in [1,2,3]],
    prevent_initial_call=True
)
def update(button, date, tree, measurement):
    global TEMP_CSV_FILE
    _ = date.split("-")
    _.reverse()
    date = "".join(_)
    file = f"../01_Mereni_Babice_{date}_optika_zpracovani/csv/{tree}_{measurement}.csv"
    df = pd.read_csv(file,
                  header=[0,1], index_col=0, dtype = np.float64)    
    df.index=df["Time"].values.reshape(-1)
    data = df[["Time","Pt3","Pt4","Pt11","Pt12","Pt13"]]
    data.to_csv(TEMP_CSV_FILE)
    return f"file {file} loaded",{},{},None,None

# Nakresli se graf, bud kliknutim na tlacitko nebo prepnutim probu
@callback(
    Output('result', 'children', allow_duplicate=True),
    Output('graph-content', 'figure', allow_duplicate=True),
    Output('graph-content-2', 'figure', allow_duplicate=True),
    Output('start', 'children', allow_duplicate=True),
    Output('end', 'children', allow_duplicate=True),
    Input('plot-button', 'n_clicks'),
    Input('radio-selection-4', 'value'),
    *[State(f'radio-selection-{i}', 'value') for i in [1,2,3]],
    prevent_initial_call=True
)
def update_fig(button, probe, day, tree, measurement):
    global TEMP_CSV_FILE
    # global CSV_FILE_PEAKS
    # global X_AXES
    # global Y_AXES
    if probe is None:
        raise PreventUpdate()
    data = pd.read_csv(TEMP_CSV_FILE, header=[0,1], index_col=0, dtype = np.float64)
    data = data[[i for i in data.columns if "X" not in i[1]]]
    data.columns = [i[0] for i in data.columns]
    data = data - data.iloc[0,:]
    fig = go.Figure(layout_title_text=f"{day} {tree} {measurement}: {probe}")
    fig.add_trace(go.Scatter(x=data.index, y=data[probe], mode='lines', name=probe))
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Delta position")
    return "Graph finished",fig,{},None,None


# Pokud uživatel vybral část grafu, upraví se počáteční a koncová hodnota
# času pro FFT
@callback(
    Output('start', 'children'),
    Output('end', 'children'),
    Output('result', 'children', allow_duplicate=True),
    Input('graph-content', 'relayoutData'),
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
    # Input('graph-content', 'relayoutData'),
    Input('start', 'children'),
    Input('end', 'children'),
    prevent_initial_call=True    
    )
def update_fft(start, end):
    if start is None:
        return {}
    df = pd.read_csv("temp/data.csv",
                  header=[0,1], index_col=0, dtype = np.float64)    
    limits = f"FFT from {start:.2f} to {end:.2f}"
    df = df[start:end]
    data=df.iloc[:,0].values
    time=df.index
    xf,yf = do_fft(data,time)
    fig2 = px.scatter(x=xf, y=yf, 
                    labels={'x': "Frekvence", 'y': 'Amplituda'}, 
                    log_y=True, range_x=[0,5],
                    title=limits)
    return fig2

    
if __name__ == '__main__':
    adresar = "temp"
    if not os.path.exists(adresar):
        os.makedirs(adresar)    
    app.run(debug=True)
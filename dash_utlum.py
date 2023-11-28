#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 02:14:14 2023

@author: marik
"""

from dash import Dash, html, dcc, callback, Output, Input, callback_context
from dash.dash import no_update
from dash.exceptions import PreventUpdate
import plotly.express as px
# import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
# from lib_dynatree import do_fft
import os

pio.templates.default = "plotly_white"

# Initialize the app - incorporate a Dash Bootstrap theme
# external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

days = ['2022-04-05','2022-08-16','2021-03-22','2021-06-29']
days.sort()

trees = ["01", "04", "07", "08", "09", "10", "11", "12", "13", "14", "16", "21", "24"]
trees = ["BK"+i for i in trees]

measurements = ["M02", "M03", "M04", "M05", "M06"]

probes = ["Pt3","Pt4"]

fixed_by_probes = ["None","Pt11","Pt12", "Pt13"]

app.layout =  dbc.Container(
    [
    html.H1('Měření Babice' ),
    dbc.Row(
    [dbc.Col(i) for i in [
        ["Day",dcc.RadioItems(days, value=days[0], id='radio-selection-1', labelStyle={'margin-right': '20px'}, inline=True)],
        ["Tree",dcc.RadioItems(trees, value=trees[0], id='radio-selection-2', labelStyle={'margin-right': '20px'}, inline=True)],
        ["Measurement",dcc.RadioItems(measurements, value=measurements[0], id='radio-selection-3', labelStyle={'margin-right': '20px'}, inline=True)],
        ["Probe to plot",dcc.RadioItems(probes, id='radio-selection-4', labelStyle={'margin-right': '20px'}, inline=True)],
        # ["Probe to fix the movement",dcc.RadioItems(fixed_by_probes, value=fixed_by_probes[0], id='radio-selection-5', labelStyle={'margin-right': '20px'}, inline=True)],
    ]]),
    dbc.Button('Load CSV', id='load-button'),
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
CSV_FILE_PEAKS = "csv/peaks_for_decrement.csv"

# Podle radioboxu se načtou data a uloží se studované sloupce 
# do menší databáze.
@callback(
    Output('result', 'children',  allow_duplicate=True),
    Output('graph-content', 'figure', allow_duplicate=True),
    Output('graph-content-2', 'figure', allow_duplicate=True),
    Output('start', 'children', allow_duplicate=True),
    Output('end', 'children', allow_duplicate=True),
    *[Input(f'radio-selection-{i}', 'value') for i in [1,2,3]],
    Input('load-button', 'n_clicks'),
    prevent_initial_call=True
)
def update(date, tree, measurement, button):
    global TEMP_CSV_FILE
    _ = date.split("-")
    _.reverse()
    date = "".join(_)
    file = f"../01_Mereni_Babice_{date}_optika_zpracovani/csv/{tree}_{measurement}.csv"
    if callback_context.triggered_id != 'load-button':
        return f"Click Load to load the file {file}",{},{},None,None
    try:
        df = pd.read_csv(file,
                 header=[0,1], index_col=0, dtype = np.float64)    
    except:
        return f"Soubor {file} neexistuje",{},{},None,None
    df.index=df["Time"].values.reshape(-1)
    data = df[["Time","Pt3","Pt4","Pt11","Pt12","Pt13"]]
    data.to_csv(TEMP_CSV_FILE)
    return f"file {file} loaded",{},{},None,None

@callback(
    Output('result', 'children', allow_duplicate=True),
    Output('graph-content', 'figure', allow_duplicate=True),
    Output('start', 'children', allow_duplicate=True),
    Output('end', 'children', allow_duplicate=True),
    Input('radio-selection-4', 'value'),
    prevent_initial_call=True
)
def update_fig(probe):
    global TEMP_CSV_FILE
    data = pd.read_csv(TEMP_CSV_FILE, header=[0,1], index_col=0, dtype = np.float64)
    data = data - data.iloc[0,:]
    output_data = data.loc[:,[(i,"Y0") for i in [probe,"Pt11","Pt12","Pt13"]]]
    for i in ["Pt11","Pt12","Pt13"]:
        output_data[i] = output_data[i]+output_data[probe]
    output_data.columns = [f"{probe}{i}" for i in ["", " fixed by Pt11"," fixed by Pt12"," fixed by Pt13"]]
    labels = {'index': "Time", 'value': "Position"}
    fig = px.line(output_data, labels=labels, title=f"{probe} movement")
    return "Graph finished",fig,None,None



# Pokud uživatel vybral část grafu, upraví se počáteční a koncová hodnota
# času pro FFT
@callback(
    Output('result', 'children', allow_duplicate=True),
    Input('graph-content', 'clickData'),
    *[Input(f'radio-selection-{i}', 'value') for i in [1,2,3,4]],
    prevent_initial_call=True
    )
def save_point_on_graph(graph, day, tree, measurement, probe):
    global CSV_FILE_PEAKS
    if graph is None:
        raise PreventUpdate()
    var_names = [f"{probe}", f"{probe} fixed by Pt11", f"{probe} fixed by Pt12", f"{probe} fixed by Pt13"]
    point = graph['points'][0]
    coordinates = (point['x'],point['y'])
    var_name = var_names[point['curveNumber']]
    df_new = pd.DataFrame({'day': [day], 'tree': [tree], 'measurement': [measurement], 'variable': [var_name], 
                        'time': [point['x']], 'value': [point['y']]})
    try:
        df_old = pd.read_csv(CSV_FILE_PEAKS, index_col=None)
        df = pd.concat([df_old,df_new])
    except:
        df = df_new
    df.to_csv(CSV_FILE_PEAKS, index=False)
    # return f"{point['x']}, {point['y']}"
    return f"Clicked {var_name} at time {coordinates[0]}, value {coordinates[1]}, saved to {CSV_FILE_PEAKS}"


# # FFT
# @callback(
#     Output('graph-content-2', 'figure', allow_duplicate=True),
#     # Input('graph-content', 'relayoutData'),
#     Input('start', 'children'),
#     Input('end', 'children'),
#     prevent_initial_call=True    
#     )
# def update_fft(start, end):
#     if start is None:
#         return {}
#     df = pd.read_csv("temp/data.csv",
#                   header=[0,1], index_col=0, dtype = np.float64)    
#     limits = f"FFT from {start:.2f} to {end:.2f}"
#     df = df[start:end]
#     data=df.iloc[:,0].values
#     time=df.index
#     xf,yf = do_fft(data,time)
#     fig2 = px.scatter(x=xf, y=yf, 
#                     labels={'x': "Frekvence", 'y': 'Amplituda'}, 
#                     log_y=True, range_x=[0,5],
#                     title=limits)
#     return fig2

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
    try:
        os.remove(TEMP_CSV_FILE)
    except:
        pass
    app.run(debug=True)
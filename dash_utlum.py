#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 02:14:14 2023

@author: marik
"""

from dash import Dash, html, dcc, callback, Output, Input, State
from dash.dash import no_update
from dash.exceptions import PreventUpdate
# import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import lib_analyze_filenames as laf
# from lib_dynatree import do_fft
import os

pio.templates.default = "plotly_white"

# Initialize the app - incorporate a Dash Bootstrap theme
# external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

# import pandas as pd


probes = [
    "Pt3", "Pt3 fixed by Pt11", "Pt3 fixed by Pt12", "Pt3 fixed by Pt13",
    "Pt4", "Pt4 fixed by Pt11", "Pt4 fixed by Pt12", "Pt4 fixed by Pt13",
    ]

fixed_by_probes = ["None","Pt11","Pt12", "Pt13"]

app.layout =  dbc.Container(
    [
    html.H1('Měření Babice' ),
    dbc.Row(
    [dbc.Col(i) for i in [
        ["Day",dcc.RadioItems(laf.days, id='radio-selection-1', labelStyle={'margin-right': '20px'}, inline=True)],
        ["Tree",dcc.RadioItems([], id='radio-selection-2', labelStyle={'margin-right': '20px'}, inline=True)],
        ["Measurement",dcc.RadioItems([] , id='radio-selection-3', labelStyle={'margin-right': '20px'}, inline=True)],
        ["Probe to plot",dcc.RadioItems(probes, value=None, id='radio-selection-4', labelStyle={'margin-right': '20px'})],
    ]]),
    dbc.Button('Load CSV', id='load-button'),html.Span(" "),
    dbc.Button('Plot', id='plot-button'),html.Span(" "),
    dbc.Button('Delete manual peaks', id='delete-button'),    
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
X_AXES = [None,None]
Y_AXES = [None,None]


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
    Output('start', 'children', allow_duplicate=True),
    Output('end', 'children', allow_duplicate=True),
    Input('plot-button', 'n_clicks'),
    Input('radio-selection-4', 'value'),
    *[State(f'radio-selection-{i}', 'value') for i in [1,2,3]],
    prevent_initial_call=True
)
def update_fig(button, probe, day, tree, measurement):
    global TEMP_CSV_FILE
    global CSV_FILE_PEAKS
    global X_AXES
    global Y_AXES
    if probe is None:
        raise PreventUpdate()
    data = pd.read_csv(TEMP_CSV_FILE, header=[0,1], index_col=0, dtype = np.float64)
    data = data[[i for i in data.columns if "X" not in i[1]]]
    data.columns = [i[0] for i in data.columns]
    data = data - data.iloc[0,:]
    if "fixed" in probe:
        data[probe] = data[probe[:3]] - data[probe[-4:]]
    fig = go.Figure(layout_title_text=f"{day} {tree} {measurement}: {probe}")
    fig.add_trace(go.Scatter(x=data.index, y=data[probe], mode='lines', name=probe))
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Delta position")
    if X_AXES[0] is not None:
        fig.update_xaxes(range=X_AXES)
    if Y_AXES[0] is not None:
        fig.update_yaxes(range=Y_AXES)
    try:
        df_peaks = pd.read_csv(CSV_FILE_PEAKS, index_col=[0,1,2,3]).sort_index()
        peaks = df_peaks.loc[(day,tree,measurement,probe)].values
        fig.add_trace(go.Scatter(x=peaks[:,0], y=peaks[:,1], mode='markers', name='markers', marker_size=10))
    except:
        pass
    # fig = px.line(output_data, labels=labels, title=f"{probe} movement")
    # fig.add_trace(px.scatter(x=[0,10,20],y=[1,2,3], mode='markers'))
    return "Graph finished",fig,None,None

# Pokud uživatel vybral část grafu, upraví se globální proměnné

@callback(
    Output('result', 'children', allow_duplicate=True),
    Input('graph-content', 'relayoutData'),
    prevent_initial_call=True
    )
def update_fft_bounds(graph):
    global X_AXES
    global Y_AXES
    if graph is not None and "xaxis.range[0]" in graph.keys():
        X_AXES[0] = graph["xaxis.range[0]"]
    if graph is not None and "xaxis.range[1]" in graph.keys():
        X_AXES[1]= graph["xaxis.range[1]"]
    if graph is not None and "yaxis.range[0]" in graph.keys():
        Y_AXES[0] = graph["yaxis.range[0]"]
    if graph is not None and "yaxis.range[1]" in graph.keys():
        Y_AXES[1]= graph["yaxis.range[1]"]
    return f"Limits updated to {graph}."


@callback(
    Output('result', 'children', allow_duplicate=True),
    Input('delete-button', 'n_clicks'),
    *[State(f'radio-selection-{i}', 'value') for i in [1,2,3,4]],
    prevent_initial_call=True
)
def delete_peaks(button, day, tree, measurement, probe):
    global CSV_FILE_PEAKS
    df = pd.read_csv(CSV_FILE_PEAKS,index_col=(0,1,2,3))
    df.drop((day,tree,measurement,probe), inplace=True)
    df.to_csv(CSV_FILE_PEAKS)
    return f"Deleted manual peaks for {day}-{tree}-{measurement} from {CSV_FILE_PEAKS}"

@callback(
    Output('result', 'children', allow_duplicate=True),
    Input('graph-content', 'clickData'),
    *[State(f'radio-selection-{i}', 'value') for i in [1,2,3,4]],
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
    return f"Clicked {var_name} at time {coordinates[0]}, value {coordinates[1]}, saved to {CSV_FILE_PEAKS}"

# # @callback(
# #     Output('placeholder', 'children'),
# #     *[Input(f'radio-selection-{i}', 'value') for i in [1,2,3,4]],
# #     Input('start', 'children'),
# #     Input('end', 'children'),
# #     Input('button', 'n_clicks'),
# #     prevent_initial_call = True
# #     )
# # def update_output(date, tree, measurement, probe, start, end, btn):
# #     if "button" == ctx.triggered_id:
# #         return f'From {start} to {end}.'
# #     else:
# #         raise PreventUpdate
    

if __name__ == '__main__':
    adresar = "temp"
    if not os.path.exists(adresar):
        os.makedirs(adresar)    
    try:
        os.remove(TEMP_CSV_FILE)
    except:
        pass
    app.run(debug=True)
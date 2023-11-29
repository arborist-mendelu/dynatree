#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 02:14:14 2023

@author: marik
"""

from dash import Dash, html, dcc, callback, Output, Input, callback_context, State
from dash.dash import no_update
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
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



import glob
import pandas as pd

def file2data(filename):
    filename = filename.split("_")
    datum = filename[3]
    datum = f"{datum[-4:]}-{datum[2:4]}-{datum[:2]}"
    return datum,filename[5].split("/")[-1],filename[6].split(".")[0]    
  
csv_files = glob.glob("../01_Mereni_Babice*optika_zpracovani/csv/*.csv")
csv_files.sort()
csv_files = [file2data(i) for i in csv_files]
df = pd.DataFrame(csv_files, columns=["date","tree", "measurement"])
df["date_tree"] = df["date"]+"_"+df["tree"]

days = df["date"].drop_duplicates().values
days.sort()
days2trees = {}
for day in days:
    days2trees[day] = df[["date","tree"]].drop_duplicates().query('date==@day')["tree"].values

days_trees = df["date_tree"].drop_duplicates().values
day_tree2measurements = {}
for day_tree in days_trees:
    day_tree2measurements[day_tree] = df[["date_tree","measurement"]].drop_duplicates().query('date_tree==@day_tree')["measurement"].values

probes = ["Pt3","Pt4"]

fixed_by_probes = ["None","Pt11","Pt12", "Pt13"]

app.layout =  dbc.Container(
    [
    html.H1('Měření Babice' ),
    dbc.Row(
    [dbc.Col(i) for i in [
        ["Day",dcc.RadioItems(days, id='radio-selection-1', labelStyle={'margin-right': '20px'}, inline=True)],
        ["Tree",dcc.RadioItems([], id='radio-selection-2', labelStyle={'margin-right': '20px'}, inline=True)],
        ["Measurement",dcc.RadioItems([] , id='radio-selection-3', labelStyle={'margin-right': '20px'}, inline=True)],
        ["Probe to plot",dcc.RadioItems(probes, value=probes[0], id='radio-selection-4', labelStyle={'margin-right': '20px'}, inline=True)],
        # ["Probe to fix the movement",dcc.RadioItems(fixed_by_probes, value=fixed_by_probes[0], id='radio-selection-5', labelStyle={'margin-right': '20px'}, inline=True)],
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
def vymaz_docasny_csv():
    global TEMP_CSV_FILE
    try:
        os.remove(TEMP_CSV_FILE)
    except:
        pass
    return None


@callback(
    Output('radio-selection-2', 'options', allow_duplicate=True),
    Output('radio-selection-3', 'options', allow_duplicate=True),
    Output('graph-content', 'figure', allow_duplicate=True),
    Input('radio-selection-1', 'value'),
    prevent_initial_call=True
    )
def nastav_stromy(datum):
    vymaz_docasny_csv()
    return days2trees[datum],[],{}


@callback(
    Output('radio-selection-3', 'options', allow_duplicate=True),
    Output('graph-content', 'figure', allow_duplicate=True),
    Input('radio-selection-2', 'value'),
    State('radio-selection-1', 'value'),
    prevent_initial_call=True
    )
def nastav_mereni(strom,datum):
    vymaz_docasny_csv()
    return day_tree2measurements[f"{datum}_{strom}"],{}

@callback(
    Output('graph-content', 'figure', allow_duplicate=True),
    Input('radio-selection-3', 'value'),
    prevent_initial_call=True
    )
def vymaz_graf(mereni):
    vymaz_docasny_csv()
    return {}


# Podle radioboxu se načtou data a uloží se studované sloupce 
# do menší databáze.
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

@callback(
    Output('result', 'children', allow_duplicate=True),
    Output('graph-content', 'figure', allow_duplicate=True),
    Output('start', 'children', allow_duplicate=True),
    Output('end', 'children', allow_duplicate=True),
    Input('plot-button', 'n_clicks'),
    *[State(f'radio-selection-{i}', 'value') for i in [1,2,3,4]],
    prevent_initial_call=True
)
def update_fig(button, day, tree, measurement, probe):
    global TEMP_CSV_FILE
    global CSV_FILE_PEAKS
    data = pd.read_csv(TEMP_CSV_FILE, header=[0,1], index_col=0, dtype = np.float64)
    data = data - data.iloc[0,:]
    output_data = data.loc[:,[(i,"Y0") for i in [probe,"Pt11","Pt12","Pt13"]]]
    for i in ["Pt11","Pt12","Pt13"]:
        output_data[i] = output_data[i]+output_data[probe]
    output_data.columns = [f"{probe}{i}" for i in ["", " fixed by Pt11"," fixed by Pt12"," fixed by Pt13"]]
    labels = {'index': "Time", 'value': "Position"}
    fig = go.Figure()
    for col in output_data.columns:
        fig.add_trace(go.Scatter(x=output_data.index, y=output_data[col], mode='lines', name=col))
        
    try:
        df_peaks = pd.read_csv(CSV_FILE_PEAKS, index_col=[0,1,2,3]).sort_index()
        peaks = df_peaks.loc[(day,tree,measurement)].values
        fig.add_trace(go.Scatter(x=peaks[:,0], y=peaks[:,1], mode='markers', name='markers', marker_size=15))
    except:
        pass
    # fig = px.line(output_data, labels=labels, title=f"{probe} movement")
    # fig.add_trace(px.scatter(x=[0,10,20],y=[1,2,3], mode='markers'))
    return "Graph finished",fig,None,None


@callback(
    Output('result', 'children', allow_duplicate=True),
    Input('delete-button', 'n_clicks'),
    *[State(f'radio-selection-{i}', 'value') for i in [1,2,3]],
    prevent_initial_call=True
)
def delete_peaks(button, day, tree, measurement):
    global CSV_FILE_PEAKS
    df = pd.read_csv(CSV_FILE_PEAKS,index_col=(0,1,2))
    df.drop((day,tree,measurement), inplace=True)
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
    # return f"{point['x']}, {point['y']}"
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
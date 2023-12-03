#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 17:34:22 2023

@author: marik
"""

from csv_add_inclino import extend_one_csv
from plot_probes_inclino_force import plot_one_measurement
import lib_analyze_filenames as laf
from lib_dash import csv_selection
from dash import Dash, html, callback, Output, Input, State, dcc, ctx
from dash.exceptions import PreventUpdate
import pandas as pd
import dash_bootstrap_components as dbc
import matplotlib.pyplot as plt
from lib_dynatree import read_data_selected

import io
import base64

app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])

app.layout =  dbc.Container(
    [
    html.H1('Měření Babice, synchronizace a pre-release data', className="text-primary text-center fs-3"),
    *csv_selection(),
    dbc.Alert([
    html.Span("Nastavit meze"),
    html.Span([dcc.RangeSlider(0, 1, value=[0,1], id='slider',
                    tooltip={"placement": "bottom", "always_visible": True})],
                    style={'flex-grow': '1'}),    
    dbc.Button('Replot', id='plot-button'),
    ], style={'display': 'flex'}),
    dbc.Alert([
    html.Span("Nastavit DPI"),
    html.Div([
    dcc.Slider(50, 300, value=100, id='slider-dpi',tooltip={"placement": "bottom", "always_visible": True}),    
    ],style={'flex-grow': '1'})
    ], style = {'display': 'flex', 'margin': 'solid'}, color="success"),
    dcc.Loading(
    id="loading-1",
    type="default",
    children=html.Img(id='image'), # img element
    ),
    dcc.Markdown("""
* vyberte si den, strom a měření, vykreslí se kmity Pt3 nahoře, síla, elastometr, inklinometry
* zkontroluj, jestli je správně zarovnán okamžik vypuštění, kontroluj v posledním obrázku sílu a výchylku.
* pokud síla a výchylka nejsou zarovnány okamžikem vypuštění, můžeš doladit v souboru 
  `csv/synchronization_finetune_inclinometers_fix.csv`
* po ukončení ne potřeba spustit  skript `csv_add_inclino.py` pro pro začlenění informací do `csv_extra`,
  dále `extract_release_data.py` pro opravená data před vypuštěním a případně `plot_probes_inclino_force.py`
  pro obrázky jaké jsou zde.
                 """
                 ),
    ])
DF = pd.DataFrame()

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

FILENAME = ""  # the last csv file which has been loaded and drawn

# Zmena mereni ulozi jmeno souboru a nacte dataframe
@callback(
    Output('csv', 'children', allow_duplicate=True),
    Output('slider', 'max', allow_duplicate=True),
    Output('slider', 'value', allow_duplicate=True),
    Output('image', 'src', allow_duplicate=True),
    Input('radio-selection-3', 'value'),
    *[State(f'radio-selection-{i}', 'value') for i in [1,2]],
    prevent_initial_call=True
    )
def sestav_csv(measurement, date, tree):
    global DF
    global FILENAME
    if measurement is None or date is None or tree is None:
        return None,1,[0,1],None
    _ = date.split("-")
    _ .reverse()
    _ = "".join(_)
    file = f"../01_Mereni_Babice_{_}_optika_zpracovani/csv/{tree}_{measurement}.csv"
    DF = read_data_selected(file)
    FILENAME = file
    return file,DF.index.max(),[0,DF.index.max()],None

@callback(
    Output('image', 'src', allow_duplicate=True),
    Input('csv', 'children'),
    Input('plot-button', 'n_clicks'),
    *[State(f'radio-selection-{i}', 'value') for i in [1,2,3]],
    State('slider','value'),
    State('slider-dpi','value'),
    prevent_initial_call=True
    )    
def plot_graph(file, button, day, tree, measurement,slider,dpi):
    global DF
    if day is None or tree is None or measurement is None:
        return None
        
    buf = io.BytesIO() # in-memory files
    
    df_ext = extend_one_csv(measurement_day=day, 
            tree=tree, 
            tree_measurement=measurement, 
            path="../", 
            write_csv=False,
            df=DF)    
    df_ext["Time"] = df_ext.index
    xlim = (None, None)
    if "plot-button" == ctx.triggered_id:
        xlim = slider
    fig = plot_one_measurement(
            measurement_day=day,
            tree=tree, 
            tree_measurement=measurement, 
            xlim=xlim,
            df_extra=df_ext,
            df=DF, 
            figsize=(10,8))    
    plt.savefig(buf, format = "png", dpi=dpi)
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode('utf-8') # encode to html elements
    buf.close()
    return "data:image/png;base64,{}".format(data)    

# # %%
# measurement = "M03"
# tree = "BK04"
# day = "2021-03-22"
# DF = read_data("../01_Mereni_Babice_22032021_optika_zpracovani/csv/BK04_M03.csv")
# # %%

# df_ext = extend_one_csv(measurement_day=day, 
#         tree=tree, 
#         tree_measurement=measurement, 
#         path="../", 
#         write_csv=False,
#         df=DF)  
# # %%
# df_ext["Time"] = df_ext.index
# plot_one_measurement(
#         measurement_day=day,
#         tree=tree, 
#         tree_measurement=measurement, 
#         xlim=(None,None),
#         df_extra=df_ext,
#         df=DF
#         ) 
# %%
# df_ext = extend_one_csv(measurement_day=day, 
#         tree=tree, 
#         tree_measurement=measurement, 
#         path="../", 
#         write_csv=False)

# plot_one_measurement(
#         measurement_day=day,
#         tree=tree, 
#         tree_measurement=measurement, 
#         # xlim=(35,42),
#         df_extra=None)

if __name__ == '__main__':
    # app.run(debug=True)
    app.run()
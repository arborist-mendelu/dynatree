#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 15:34:33 2023

Knihovna pro Dash a DYNATREE.

Obsahuje radiobuttony pro vyber data, stromu a mereni.

@author: marik
"""



from dash import  html, dcc
import dash_bootstrap_components as dbc
import lib_analyze_filenames as laf

def csv_selection(probes=None):
    probelist = []
    if probes is not None:
        probelist = ["Probe",dcc.RadioItems(probes, value=probes[0], id='radio-selection-4', labelStyle={'margin-right': '20px'}, inline=True)]
    return [  dbc.Row([
        *[dbc.Col(i) for i in [
        ["Day",dcc.RadioItems(laf.days, id='radio-selection-1', labelStyle={'margin-right': '20px'}, inline=True)],
        ["Tree",dcc.RadioItems([], id='radio-selection-2', labelStyle={'margin-right': '20px'}, inline=True)],
        ["Measurement",dcc.RadioItems([], id='radio-selection-3', labelStyle={'margin-right': '20px'}, inline=True)],
        probelist,
        ]]]),
        html.P(id='csv', style={'color': 'gray'}),
        html.P(id='message', style={'color': 'gray'}),
    ]


def make_graph(i):
    return dcc.Loading(
                id=f"ls-loading-{i}",
                children=[dcc.Graph(id=f'graph-content-{i}')],
                type="circle",
            )

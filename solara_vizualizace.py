#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:00:04 2024

@author: marik
"""

from static_pull import get_all_measurements, available_measurements
import lib_dynatree
import solara.express as px
import solara.lab
import solara
import pandas as pd
DATA_PATH = "../data"

tightcols = {'gap': "0px"}
regression_settings = {'color': 'gray', 'alpha': 0.5}


title = "DYNATREE: vizualizace dat, se kterými se pracuje"


methods = solara.reactive(['normal', 'den', 'noc', 'afterro', 'mraz'])
method = solara.reactive('normal')
widths = [800,1000,1200,1400,1600,1800]
width = solara.reactive(1200)
heights = [400,600,800,1000,1200,1400]
height = solara.reactive(600)
show_data = solara.reactive(False)

df = solara.reactive(get_all_measurements(method=method.value))
days = solara.reactive(df.value["date"].drop_duplicates().values)
trees = solara.reactive(df.value["tree"].drop_duplicates().values)
measurements = solara.reactive(df.value["measurement"].drop_duplicates().values)


def get_measuerements_list(x='all'):
    df.value = get_all_measurements(method='all', type=x)
    days.value = df.value["date"].drop_duplicates().values
    trees.value = df.value["tree"].drop_duplicates().values
    measurements.value = df.value["measurement"].drop_duplicates().values

day = solara.reactive(days.value[0])
tree = solara.reactive(trees.value[0])
measurement = solara.reactive(measurements.value[0])

data_object = lib_dynatree.DynatreeMeasurement(
    day.value, 
    tree.value, 
    measurement.value,
    measurement_type=method.value,
    )

dependent_pull = solara.reactive(["Force(100)"])
dependent_pt34 = solara.reactive(["Pt3"])
dependent_extra = solara.reactive(["Force(100)"])
# 
def resetuj_a_nakresli(x=None):
    pass

def nakresli(x=None):
    pass

def investigate(df_, var, msg=None):
    df = df_.copy()
    solara.ToggleButtonsMultiple(value=var, values=list(df.columns))    
    px.scatter(df, y=var.value,  height = height.value, width=width.value, **kwds)    
    if msg is not None:
        msg
    number_nans = df.isna().sum()
    is_nan = number_nans.sum() == 0
    number_nans = pd.DataFrame(number_nans).T
    number_nans.index = ["# of Nan values"]
    with solara.Info():
        if is_nan:
            solara.Text("There are no undefined values in the dataframe.")
        else:
            solara.display(number_nans)
    if show_data.value:
        df["Time"] = df.index
        solara.DataFrame(df)

def redraw(x=None):
    pass
    
kwds = {"template": "plotly_white", 
        # "height": height.value, "width": width.value
        }

@solara.component
def Page():
    data_object = lib_dynatree.DynatreeMeasurement(
        day.value, 
        tree.value, 
        measurement.value,
        measurement_type=method.value,
        )
    solara.Title(title)
    solara.Style(".widget-image{width:100%;} .v-btn-toggle{display:inline;}  .v-btn {display:inline; text-transform: none;} .vuetify-styles .v-btn-toggle {display:inline;} .v-btn__content { text-transform: none;}")
    with solara.Sidebar():
        Selection()
    with solara.lab.Tabs():
        with solara.lab.Tab("Tahovky"):
            with solara.Card():
                try:
                    df = data_object.data_pulling
                    investigate(df, dependent_pull)
                except:
                    pass
        with solara.lab.Tab("Optika Pt3 a Pt4"):
            with solara.Card():
                try:
                    if data_object.is_optics_available:
                        df2 = data_object.data_optics_pt34
                        df2 = df2-df2.iloc[0,:]
                        df2.columns = [i[0] for i in df2.columns]
                        df2 = df2[[i for i in df2.columns if "Time" not in i]]
                        investigate(df2, dependent_pt34, msg=solara.Info(solara.Markdown("The data are shifted to start from zero.")))                        
                        pass
                    else:
                        solara.Warning(solara.Markdown("Optika pro toto měření není dostupá. Buď neexistuje, nebo ještě není zpracovaná."))
                # try:
                    # Detail()
                except:
                    pass
        with solara.lab.Tab("Tahovky interpolovane na optiku"):
            with solara.Card():
                try:
                    if data_object.is_optics_available:
                        df3 = data_object.data_optics_extra
                        # df3 = df3-df3.iloc[0,:]
                        df3.columns = [i[0] for i in df3.columns]
                        df3 = df3[[i for i in df3.columns if "fixed" not in i]]
                        investigate(df3, dependent_extra)
                        pass
                    else:
                        solara.Warning(solara.Markdown("Optika pro toto měření není dostupá. Buď neexistuje, nebo ještě není zpracovaná."))
                # try:
                    # Detail()
                except:
                    pass

def Selection():
    with solara.Card(title="Measurement choice"):
        with solara.Column():
            solara.ToggleButtonsSingle(value=method, values=list(methods.value),
                                       on_value=get_measuerements_list)
            solara.ToggleButtonsSingle(value=day, values=list(days.value),
                                       on_value=resetuj_a_nakresli)
            solara.ToggleButtonsSingle(value=tree, values=list(trees.value),
                                       on_value=resetuj_a_nakresli)
            solara.ToggleButtonsSingle(value=measurement,
                                       values=available_measurements(
                                           df.value, day.value, tree.value, method.value),
                                       on_value=nakresli
                                       )
        data_object = lib_dynatree.DynatreeMeasurement(
            day.value, tree.value, measurement.value,measurement_type=method.value)
        solara.Markdown(
            f"**Selected**: {day.value}, {tree.value}, {measurement.value}")
        if data_object.is_optics_available:
            solara.Markdown("✅ Optics is available for this measurement.")
        else:
            solara.Markdown(
                "❎ Optics is **not** available for this measurement.")
        with solara.Tooltip("Allows to see the data table. Default is off, i.e. save the bandwidth."):
            solara.Switch(
                label="Show data table",
                value=show_data,
            )            
            
    with solara.Card("Image setting"):
        with solara.Column():
            solara.Markdown("Image width")
            solara.ToggleButtonsSingle(value=width, values=widths)
            solara.Markdown("Image height")
            solara.ToggleButtonsSingle(value=height, values=heights)
      


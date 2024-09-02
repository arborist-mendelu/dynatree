#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:00:04 2024

@author: marik
"""

from lib_find_measurements import get_all_measurements, available_measurements
import lib_dynatree
import plotly.express as px
import solara.lab
import solara
import pandas as pd
import time

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


def get_measurements_list(x='all'):
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
dependent_bl = solara.reactive(["BL44"])
dependent_extra = solara.reactive(["Force(100)"])

def list_drop():
    list_selected.value = pd.DataFrame()

def first_drop():
    list_selected.value = list_selected.value.iloc[1:,:].reset_index(drop=True)
    
list_selected = solara.reactive(pd.DataFrame())
auto_add = solara.reactive(False)

@solara.component
def investigate(df_, var, msg=None):
    selection_data, set_selection_data = solara.use_state(None)
    
    
    def current_selection(selection_data):
        ans = {
            'measurement_type': [method.value],
            'day':[day.value], 
            'tree':[tree.value], 
            'measurement':[measurement.value], 
            'probe':[var.value[0]],
            'xmin':[selection_data['selector']['selector_state']['xrange'][0]],
            'xmax':[selection_data['selector']['selector_state']['xrange'][1]],
            'ymin':[selection_data['selector']['selector_state']['yrange'][0]],
            'ymax':[selection_data['selector']['selector_state']['yrange'][1]],
            }
        ans = pd.DataFrame(ans)
        return ans
        
    def save_data():
        data = current_selection(selection_data)
        if list_selected.value.shape[0]==0:
            list_selected.value = data
            return
        equal_dataset = list_selected.value.iloc[0,:5].equals(data.iloc[0,:5])
        if equal_dataset:
            list_selected.value.iloc[0,:] = data.iloc[0,:]
        else:
            df = pd.concat([
                data,
                list_selected.value
                ]).reset_index(drop=True)
            list_selected.value = df
    
    df = df_.copy()
    solara.ToggleButtonsMultiple(value=var, values=list(df.columns))    
    fig = px.scatter(df, y=var.value,  height = height.value, width=width.value,
                     title=f"Dataset: {method.value}, {day.value}, {tree.value}, {measurement.value}",
                     **kwds)    
    solara.FigurePlotly(fig, on_selection=set_selection_data)
    if msg is not None:
        msg
    number_nans = df.isna().sum()
    is_nan = number_nans.sum() == 0
    number_nans = pd.DataFrame(number_nans).T
    number_nans.index = ["# of Nan values"]
    with solara.Card():
        solara.Markdown("**Undefined values (Nan, Not a number) statistics**")
        if is_nan:
            solara.Text("There are no undefined values in the dataframe.")
        else:
            solara.display(number_nans)
    if show_data.value:
        df["Time"] = df.index
        with solara.Card():
            solara.Markdown("**Data table**")
            solara.DataFrame(df)
            solara.FileDownload(df.to_csv(), filename=f"{method.value}_{day.value}_{tree.value}_{measurement.value}.csv", label="Download as csv")
    if selection_data is None:
        msg_selected = "You can use the box select tool to select some data and download the selected bounds for later use."
        save_disabled = True
    else:
        msg_selected = """
        **Current selection**
        """
        save_disabled = False 
    with solara.Card():
        solara.Markdown(msg_selected)
        if selection_data is not None:
            solara.display(current_selection(selection_data))
            if auto_add.value:
                save_data()
        solara.Markdown("**Previously selected**")
        solara.display(list_selected.value)
        with solara.Row():
            with solara.Tooltip("Add the current selection automatically on the top of previously selected. It the first five columns are equal for the current selection and the last added selection (on top), the data are rewritten to the current."):
                solara.Switch(label="AutoAdd", value=auto_add)
            solara.Button(label="Add current to table", disabled=save_disabled, on_click=save_data)
            solara.Button(label="Delete table", on_click=list_drop)
            with solara.Tooltip("Works if the switch AutoAdd is off."):
                solara.Button(label="Delete first row", on_click=first_drop)
            solara.FileDownload(list_selected.value.to_csv(), filename="data.csv", label="Download as csv")
        
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
        with solara.lab.Tab("Optika BL44-67"):
            with solara.Card():
                try:
                    if data_object.is_optics_available:
                        cols = [(f"BL{i}","Y0") for i in range(44,68)] 
                        df2 = data_object.data_optics.loc[:,cols]
                        df2 = df2-df2.iloc[0,:]
                        df2.columns = [i[0] for i in df2.columns]
                        df2 = df2[[i for i in df2.columns if "Time" not in i]]
                        investigate(df2, dependent_bl, msg=solara.Info(solara.Markdown("The data are shifted to start from zero. The value of Y0 is considered.")))                        
                    else:
                        solara.Warning(solara.Markdown("Optika pro toto měření není dostupá. Buď neexistuje, nebo ještě není zpracovaná."))
                # try:
                    # Detail()
                except:
                    pass
def resetuj_a_nakresli(x=None):
    pass

def nakresli(x=None):
    pass

def Selection():
    with solara.Card(title="Measurement choice"):
        with solara.Column():
            solara.ToggleButtonsSingle(value=method, values=list(methods.value),
                                       on_value=get_measurements_list)
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
      


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 17:01:24 2024

@author: marik
"""
import solara
from lib_find_measurements import get_all_measurements, available_measurements
import lib_dynatree

DATA_PATH = "../data"

tightcols = {'gap': "0px"}
regression_settings = {'color': 'gray', 'alpha': 0.5}

methods = solara.reactive(['normal', 'den', 'noc', 'afterro', 'afterro2', 'mraz', 'mokro'])
method = solara.reactive(methods.value[0])


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
use_optics = solara.reactive(False)
include_details = solara.reactive(False)

# Create data object when initialized
data_object = lib_dynatree.DynatreeMeasurement(
    day.value, 
    tree.value, 
    measurement.value,
    measurement_type=method.value,
    # use_optics=use_optics.value
    )

xdata = solara.reactive("M_Measure")
ydata = solara.reactive(["blue", "yellow"])
ydata2 = solara.reactive([])
pull = solara.reactive(0)

def resetuj(x=None):
    measurement.set(measurements.value[0])

@solara.component
def Selection_trees_only(tree_action = lambda x:None):
    with solara.Card(title="Tree choice"):
        vals = get_all_measurements(method='all', type='all')["tree"].drop_duplicates().values
        solara.ToggleButtonsSingle(value=tree, values=list(vals),
                                   on_value=tree_action)

@solara.component
def Selection(
       method_action = get_measurements_list,
       day_action = resetuj,
       tree_action = resetuj,
       measurement_action = lambda x:None,
       switch_action = lambda x:None,
       button_action = None,
       optics_switch = True,
       confirm_choice = True, 
       report_optics_availability = True,
       exclude_M01 = False
        ):
    if exclude_M01:
        measurements.value = [i for i in measurements.value if i!="M01"]
    with solara.Card(title="Measurement choice"):
        with solara.Column():
            # solara.Switch(label="Use data from URL", value=data_from_url)
            solara.ToggleButtonsSingle(value=method, values=list(methods.value),
                                       on_value=method_action)
            solara.ToggleButtonsSingle(value=day, values=list(days.value),
                                       on_value=day_action)
            solara.ToggleButtonsSingle(value=tree, values=list(trees.value),
                                       on_value=tree_action)
            solara.ToggleButtonsSingle(value=measurement,
                                       values=available_measurements(
                                           df.value, day.value, tree.value, method.value,
                                           exclude_M01=exclude_M01),
                                       on_value=measurement_action
                                       )
        data_object = lib_dynatree.DynatreeMeasurement(
            day.value, tree.value, measurement.value,measurement_type=method.value)
        if optics_switch:
            with solara.Tooltip("Umožní použít preprocessing udělaný na tahovkách M02 a více. Tím sice nebude stejná metodika jako pro M01 (tam se preprocessing nedělal), ale máme časovou synchronizaci s optikou, o opravu vynulování inklinometrů a pohyb bodů Pt3 a Pt4."):
                solara.Switch(
                    label="Use optics data, if possible",
                    value=use_optics,
                    disabled=not data_object.is_optics_available,
                    on_value=switch_action
                )
        # solara.Div(style={"margin-bottom": "10px"})
        if button_action is not None:
            solara.Button("Run calculation", on_click=button_action, color="primary")
        # solara.Button("Clear cache", on_click=clear(), color="primary")
        if confirm_choice:
            solara.Markdown(
                f"**Selected**: {method.value}, {day.value}, {tree.value}, {measurement.value}")
        if report_optics_availability:
            if data_object.is_optics_available:
                solara.Markdown("✅ Optics is available for this measurement.")
            else:
                solara.Markdown(
                    "❎ Optics is **not** available for this measurement.")

styles_css = """
        .widget-image{width:auto;} 
        .v-btn-toggle{display:inline;}  
        .v-btn {display:inline; text-transform: none;} 
        .vuetify-styles .v-btn-toggle {display:inline;} 
        .v-btn__content { text-transform: none;}
        """

widths = [800,1000,1200, 1400]
width = solara.reactive(1000)
heights = [400,600,800, 1000]
height = solara.reactive(600)

@solara.component
def ImageSizes():
    with solara.Card("Image setting"):
        with solara.Column():
            solara.SliderValue(label="width", value=width, values=widths)
            solara.SliderValue(label="height", value=height, values=heights)
            

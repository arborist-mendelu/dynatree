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
import lib.solara.select_source as s
import matplotlib.pyplot as plt
import matplotlib
from plotly_resampler import FigureResampler

DATA_PATH = "../data"

tightcols = {'gap': "0px"}
regression_settings = {'color': 'gray', 'alpha': 0.5}


title = "DYNATREE: vizualizace dat, se kterými se pracuje"


show_data = solara.reactive(False)

dependent_pull = solara.reactive(["Force(100)"])
dependent_pt34 = solara.reactive(["Pt3"])
dependent_bl = solara.reactive(["BL44"])
dependent_extra = solara.reactive(["Force(100)"])
dependent_acc = solara.reactive(["a01_x"])

def list_drop():
    list_selected.value = pd.DataFrame()

def first_drop():
    list_selected.value = list_selected.value.iloc[1:,:].reset_index(drop=True)
    
list_selected = solara.reactive(pd.DataFrame())
auto_add = solara.reactive(False)

selection_data = solara.reactive(None)

def set_selection_data(x=None):
    selection_data.value = x

data_object = lib_dynatree.DynatreeMeasurement(
    s.day.value, 
    s.tree.value, 
    s.measurement.value,
    measurement_type=s.method.value,
    )

@solara.component
def plot(df, var, msg=None, id=None, resample=False):
    # df = df_.copy()
    solara.ToggleButtonsMultiple(value=var, values=list(df.columns))    
    fig = px.scatter(df, y=var.value,  height = s.height.value, width=s.width.value,
                     title=f"Dataset: {s.method.value}, {s.day.value}, {s.tree.value}, {s.measurement.value}",
                     **kwds)    
    if resample:
        fig_res = FigureResampler(fig)
        solara.FigurePlotly(fig_res, on_selection=set_selection_data)    
    else:
        solara.FigurePlotly(fig, on_selection=set_selection_data)    
    if msg is not None:
        msg

# @solara.component
# def plot_img(df, var, msg=None, id=None):
#     # df = df_.copy()
#     solara.ToggleButtonsMultiple(value=var, values=list(df.columns))    
#     fig, ax = plt.subplots()
#     try:
#         df.plot(y=var.value, ax=ax)
#     except:
#         pass
#     ax.set(title=f"Dataset: {s.method.value}, {s.day.value}, {s.tree.value}, {s.measurement.value}")    
#     solara.FigureMatplotlib(fig)
#     plt.close('all')    
#     if msg is not None:
#         msg        

@solara.component
def investigate(df_, var):
    
    def current_selection():
        ans = {
            'measurement_type': [s.method.value],
            'day':[s.day.value], 
            'tree':[s.tree.value], 
            'measurement':[s.measurement.value], 
            'probe':[var.value[0]],
            'xmin':[selection_data.value['selector']['selector_state']['xrange'][0]],
            'xmax':[selection_data.value['selector']['selector_state']['xrange'][1]],
            'ymin':[selection_data.value['selector']['selector_state']['yrange'][0]],
            'ymax':[selection_data.value['selector']['selector_state']['yrange'][1]],
            }
        ans = pd.DataFrame(ans)
        return ans
        
    def save_data():
        data = current_selection()
        if list_selected.value.shape[0]==0:
            # list is empty, fill with the rist row
            list_selected.value = data
            return
        equal_dataset = list_selected.value.iloc[0,:].equals(data.iloc[0,:])
        if equal_dataset:
            # The first row already included, do nothing
            return
        equal_dataset = list_selected.value.iloc[0,:5].equals(data.iloc[0,:5])
        if equal_dataset:
            # just change in limits, no change in method, day, tree, measurement
            list_selected.value.iloc[0,:] = data.iloc[0,:]
        else:
            # put the active selection on the top
            df = pd.concat([
                data,
                list_selected.value
                ]).reset_index(drop=True)
            list_selected.value = df
    
    df = df_.copy()
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
            solara.FileDownload(df.to_csv(), filename=f"{s.method.value}_{s.day.value}_{s.tree.value}_{s.measurement.value}.csv", label="Download as csv")
    if selection_data.value is None:
        msg_selected = "You can use the box select tool to select some data and download the selected bounds for later use."
        save_disabled = True
    else:
        msg_selected = """
        **Current selection**
        """
        save_disabled = False 
    with solara.Card(style={'background-color': '#FFFFAA'}):
        solara.Markdown(msg_selected)
        if selection_data.value is not None:
            solara.display(current_selection())
            if auto_add.value:
                save_data()
    with solara.Column():
        with solara.Row(justify='center'):
            solara.Text("⇩", style={'font-size':'300%'})
            solara.Button(label="Add current selection to table", disabled=save_disabled, on_click=save_data)
            with solara.Tooltip("Add the current selection automatically on the top of previously selected. It the first five columns are equal for the current selection and the last added selection (on top), the data are rewritten to the current."):
                solara.Switch(label="AutoAdd", value=auto_add)
    with solara.Card():
        solara.Markdown("**Previously selected**")
        solara.display(list_selected.value)
        with solara.Row():
            solara.Button(label="Delete table", on_click=list_drop)
            with solara.Tooltip("Works if the switch AutoAdd is off."):
                solara.Button(label="Delete first row", on_click=first_drop)
            solara.FileDownload(list_selected.value.to_csv(), filename="data.csv", label="Download as csv")
        
kwds = {"template": "plotly_white", 
        # "height": height.value, "width": width.value
        }

tab_index = solara.reactive(0)

def resetuj(x=None):
    # Srovnani(resetuj=True)
    s.measurement.set(s.measurements.value[0])
    generuj_obrazky()
    
def generuj_obrazky(x=None):
    pass

@solara.component
def Page():
    data_object = lib_dynatree.DynatreeMeasurement(
        s.day.value, 
        s.tree.value, 
        s.measurement.value,
        measurement_type=s.method.value,
        )
    solara.Title(title)
    solara.Style(".widget-image{width:100%;} .v-btn-toggle{display:inline;}  .v-btn {display:inline; text-transform: none;} .vuetify-styles .v-btn-toggle {display:inline;} .v-btn__content { text-transform: none;}")
    with solara.Sidebar():
        s.Selection(exclude_M01=False, 
                    optics_switch=False,       
                    day_action = resetuj,
                    tree_action = resetuj,
                    measurement_action = generuj_obrazky)
        with solara.Card():
            solara.Switch(label="Show data table", value=show_data)
        s.ImageSizes()

    with solara.lab.Tabs(value=tab_index):
        with solara.lab.Tab("Tahovky"):
            with solara.Card():
                try:
                    if tab_index.value == 0:
                        major_minor = data_object.identify_major_minor
                        # solara.display(major_minor)
                        df = data_object.data_pulling
                        for i in major_minor.keys():
                            df[i] = df[major_minor[i]]
                        plot(df, dependent_pull, id="tahovky")
                        solara.Text(f"Row for csv with zeroing at given time: {s.method.value},{s.day.value},{s.tree.value},{s.measurement.value},,,,")
                        investigate(df, dependent_pull)
                except:
                    pass
        with solara.lab.Tab("Optika Pt3 a Pt4"):
            with solara.Card():
                try:
                    if (data_object.is_optics_available) and (tab_index.value==1):
                        df2 = data_object.data_optics_pt34
                        df2 = df2-df2.iloc[0,:]
                        df2.columns = [i[0] for i in df2.columns]
                        df2 = df2[[i for i in df2.columns if "Time" not in i]]
                        plot(df2, dependent_pt34, msg=solara.Info(solara.Markdown("The data are shifted to start from zero.")), id="optika")
                        investigate(df2, dependent_pt34)                        
                        pass
                    else:
                        solara.Warning(solara.Markdown("Optika pro toto měření není dostupá. Buď neexistuje, nebo ještě není zpracovaná."))
                except:
                    pass
        with solara.lab.Tab("Tahovky@optika_freq"):
            with solara.Card():
                try:
                    if (data_object.is_optics_available) and (tab_index.value==2):
                        df3 = data_object.data_optics_extra
                        # df3 = df3-df3.iloc[0,:]
                        df3.columns = [i[0] for i in df3.columns]
                        df3 = df3[[i for i in df3.columns if "fixed" not in i]]
                        plot(df3, dependent_extra,id="tahovky extra")
                        investigate(df3, dependent_extra)
                        pass
                    else:
                        solara.Warning(solara.Markdown("Optika pro toto měření není dostupá. Buď neexistuje, nebo ještě není zpracovaná. Tím pádem nejsou ani tahovky nasamplované podle optiky."))
                except:
                    pass
        with solara.lab.Tab("Optika BL44-67"):
            with solara.Card():
                try:
                    if (data_object.is_optics_available) and (tab_index.value==3):
                        cols = [(f"BL{i}","Y0") for i in range(44,68)] 
                        df2 = data_object.data_optics.loc[:,cols]
                        df2 = df2-df2.iloc[0,:]
                        df2.columns = [i[0] for i in df2.columns]
                        df2 = df2[[i for i in df2.columns if "Time" not in i]]
                        plot(df2, dependent_bl, msg=solara.Info(solara.Markdown("The data are shifted to start from zero. The value of Y0 is considered.")), id="BL")                        
                        investigate(df2, dependent_bl)                        
                    else:
                        solara.Warning(solara.Markdown("Optika pro toto měření není dostupá. Buď neexistuje, nebo ještě není zpracovaná."))
                except:
                    pass


        with solara.lab.Tab("ACC@100Hz"):
            with solara.Card():
                try:
                    if (tab_index.value==4):
                        df4 = data_object.data_acc
                        plot(df4, dependent_acc)                        
                        investigate(df4, dependent_acc)                        
                    else:
                        solara.Warning(solara.Markdown("Acc pro toto měření nejsou dostupné. Buď neexistuje, nebo ještě není zpracovaná."))
                except:
                    pass

        with solara.lab.Tab("ACC@5000Hz"):
            with solara.Card():
                try:
                    if (tab_index.value==5):
                        df5 = data_object.data_acc5000
                        solara.Warning(
"""
Data jsou dynamicky přesamplovaná pomocí plotly-resample. Mohou tedy vypadat trochu jinak než v jiném zobrazovátku, ale bez downsamplování by se s nimi 
nedalo pracovat. Downsamplování je pouze při zobrazování, nepoužívá se pro výpočty.
"""
                            )
                        plot(df5, dependent_acc, resample=True)                        
                        # plot(df5, dependent_acc)                        
                    #     investigate(df5, dependent_acc)                        
                    else:
                        solara.Warning(solara.Markdown("Data pro toto měření není dostupá. Buď neexistují, nebo ještě nejsou zpracovaná."))
                except:
                    pass
                # pass
        with solara.lab.Tab("Popis"):
            with solara.Card():
                solara.Markdown(navod)
                solara.Image("img/acc_positions.png", width="400px")

navod = """

## Akcelerometry

* Akcelerometry byly az na vyjimky davany v logickem poradi 4,1,3,2. Kdyztak v poznamakach/nakresu na disku to najdes.
* Osa ve které se tahá je Z, pak je zajímavá Y. Osa X je po vysce kmene.
* Cas je absolutni a automaticky v Matlab formatu. Ale kdybys chtel, tak tech cca 7e-5 je datum a cas a dostanes ho tusim pres timeformat. Jen si musis pohlidat, ze nezamenis mesic za den. 
* A pak se nenech zmast tim, ze (ted jen tusim, mozna je to prohozene)  acc2 x. a acc4 x,y,z jsou nahravany na jiny notas, nez zbytek akcelerometru, takze absolutni cas je kvuli tomu stejne na prd a nepresne jsme to synchronizovali (nebo chteli) podle tech tuku na zacatku zaznamu.


![](img/acc_positions.png)

"""

def resetuj_a_nakresli(x=None):
    auto_add.value = False
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
      


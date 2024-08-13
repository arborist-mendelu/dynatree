DATA_PATH = "../data"

navod = """
## Oscilace, inklinometry, elastometr

* Vyber si den, strom a měření a klikni na tlačítko. Vykreslí se kmity Pt3 nahoře, 
  inklinometry uprostřed a 
  síla a elastometr dole
* Zkontroluj, jestli je správně zarovnán okamžik vypuštění, 
  kontroluj v posledním obrázku sílu (oranžové tečky) a výchylku (modrá čára).
* Pokud síla a výchylka nejsou zarovnány okamžikem vypuštění, můžeš doladit v souboru 
  `csv/synchronization_finetune_inclinometers_fix.csv`
* Po ukončení je potřeba zohlednit změny v csv souboru. 
  Je proto potřeba spustit  skript `csv_add_inclino.py` pro pro začlenění informací do `csv_extra`,
  dále `extract_release_data.py` pro opravená data před vypuštěním a případně 
  `plot_probes_inclino_force.py` pro obrázky jaké jsou zde.
* Posuvníkem si můžeš změnit rozsah na ose x, aby šla dobře vidět kvalita nebo nekvalita
  synchronizace a aby se dalo posoudit, jestli je rozsah před vypuštěním (žlutý pás) umístěn rozumně.  

"""

import sys
import glob
import matplotlib.pyplot as plt
import lib_dynatree as lt
import pandas as pd
import numpy as np
import solara
from solara.lab import task
from csv_add_inclino import extend_one_csv
from plot_probes_inclino_force import plot_one_measurement
from lib_dynatree import read_data_selected_by_polars

def split_path(file):
    data = file.split("/")
    data[-1] = data[-1].replace(".csv","")
    return [file,data[-2].replace("_","-")] + data[-1].split("_")

def get_all_measurements(cesta=DATA_PATH):
    """
    Get dataframe with all measurements. The dataframe has columns
    date, tree and measurement.
    """
    files = glob.glob(cesta+"/csv/*/BK*.csv")        
    out = [split_path(file) for file in files]
    df = pd.DataFrame([i[1:] for i in out], columns=['day','tree', 'measurement'])
    df = df.sort_values(by=list(df.columns))
    df = df.reset_index(drop=True)
    return df
def available_measurements(df, day, tree):
    select_rows = (df["day"]==day) & (df["tree"]==tree)
    values = df[select_rows]["measurement"].values
    return list(values)

df = get_all_measurements()
days = df["day"].drop_duplicates().values
trees = df["tree"].drop_duplicates().values
measurements = df["measurement"].drop_duplicates().values
probes = ["Pt3 with Pt4","Pt3 with fixes"]

day = solara.reactive(days[0])
tree = solara.reactive(trees[0])
measurement = solara.reactive(measurements[0])
probe = solara.reactive(probes[0])
start = solara.reactive(0)
end = solara.reactive(0)

def reset_measurement(a):
    measurement.set(measurements[0])
    reset_limits(a)

def reset_limits(a):
    end.set(0)
    start.set(0)

if probe.value=="Pt3 with Pt4":
    plot_fixes = False
    plot_Pt4 = True
else:
    plot_fixes = True
    plot_Pt4 = False

@task
def nakresli():
    if measurement.value not in available_measurements(df, day.value, tree.value):
        fig, ax = plt.subplots()
        return fig
    if probe.value=="Pt3 with Pt4":
        plot_fixes = False
        plot_Pt4 = True
    else:
        plot_fixes = True
        plot_Pt4 = False
    if end.value == 0:
        endlim = None
    else:
        endlim = end.value
    file = f"{DATA_PATH}/csv/{day.value.replace('-','_')}/{tree.value}_{measurement.value}.csv"
    DF = read_data_selected_by_polars(file)
    df_ext = extend_one_csv(date=day.value, 
            tree=tree.value, 
            measurement=measurement.value, 
            path=DATA_PATH, 
            write_csv=False,
            df=DF
            )        
    fig = plot_one_measurement(
            date=day.value,
            tree=tree.value, 
            measurement=measurement.value, 
            path=DATA_PATH,
            plot_fixes=probe.value=="Pt3 with fixes", 
            plot_Pt4=probe.value=="Pt3 with Pt4",
            xlim=(start.value,endlim),
            df_extra=df_ext,
            df=DF,
            return_figure=True
            )    
    return fig
    
@solara.component
def Page():
    solara.Title("Oscillation: optics, inclinometers, elastometer, force synchro")
    with solara.Sidebar():
        solara.Markdown(navod)
    with solara.Card():
        with solara.Column():
            solara.ToggleButtonsSingle(value=day, values=list(days), on_value=reset_measurement)
            solara.ToggleButtonsSingle(value=tree, values=list(trees), on_value=reset_measurement)
            with solara.Row():
                solara.ToggleButtonsSingle(value=measurement, 
                                           values=available_measurements(df, day.value, tree.value),
                                           on_value=reset_limits)
                solara.ToggleButtonsSingle(value=probe, values=probes)
                solara.InputFloat("Start", value=start, continuous_update=False)   
                with solara.Tooltip("End of the plot. If 0, then the end of the plot is the end of the data."):             
                    solara.InputFloat("End", value=end, continuous_update=False)                
        solara.Div(style={"margin-bottom": "10px"})
        with solara.Row():
            solara.Button("Run calculation", on_click=nakresli, color="primary")
            solara.Markdown(f"**Selected**: {day.value}, {tree.value}, {measurement.value}, {probe.value}, {plot_fixes}, {plot_Pt4}")
    solara.ProgressLinear(nakresli.pending)    
    if measurement.value not in available_measurements(df, day.value, tree.value):
        solara.Error(f"Measurement {measurement.value} not available for this tree.")
        return
    
    if nakresli.finished:
        plt.show(nakresli.value)
    elif nakresli.not_called:
        solara.Text("Vyber měření a stiskni tlačítko Run calculation")
    else:
        with solara.Row():
            solara.Text("Pracuji jako ďábel. Může to ale nějakou dobu trvat.")
            solara.SpinnerSolara(size="100px")

import matplotlib.pyplot as plt
import numpy as np
import solara
import pandas as pd
import glob
import plotly.express as px
from plotly_resampler import FigureResampler
from scipy import signal

import krkoskova.lib_krkoskova as lk


data_directory = lk.data_directory

names = glob.glob(f"{data_directory}/*acc.parquet")
names = [i.replace("_acc.parquet","").replace(data_directory,"").replace("/","") for i in names]
names.sort()
names = {i.split("_")[0] : [j.split("_")[1] for j in names if i.split("_")[0] in j] for i  in names}

styles_css = """
        .widget-image{width:100%;} 
        .v-btn-toggle{display:inline;}  
        .v-btn {display:inline; text-transform: none;} 
        .vuetify-styles .v-btn-toggle {display:inline;} 
        .v-btn__content { text-transform: none;}
        """
kwds = {"template": "plotly_white"}

trees = list(names.keys())
tree = solara.reactive(trees[0])
measurement = solara.reactive("lano01")
sensor = solara.reactive("ext01")
sensors = ["ext01", "ext02"] + ['A01_z', 'A01_y', 'A02_z', 'A02_y', 'A03_z', 'A03_y']
n = solara.reactive(8)

@solara.component
def plot(df, msg=None, resample=False, title=None, xaxis="Time / s", **kw):
    # df = df_.copy()
    fig = px.line(df, title=title, labels={'x': xaxis}, **kwds, **kw)   
    fig.update_layout(
        xaxis_title=xaxis,
        yaxis_title="",
        )
    if resample:
        fig_res = FigureResampler(fig)
        solara.FigurePlotly(fig_res)    
    else:
        solara.FigurePlotly(fig)    
    if resample:
        solara.Info(
"""
Data jsou s příliš vysokou samplovací frekvencí. Kvůli plynulosti 
zobrazování jsou zde přesamplována pomocí knihovny `plotly_resampler`. 
Do výpočtů již vstupují data všechna.
""")
    if msg is not None:
        msg

@solara.component
def Page():
    solara.Style(styles_css)
    solara.Title("Krkoškova")
    with solara.Sidebar():
        solara.Markdown("**Tree**")
        solara.ToggleButtonsSingle(value=tree, values=trees)
        solara.Markdown("**Measurement**")
        solara.ToggleButtonsSingle(value=measurement, values = names[tree.value])
        solara.Markdown("**Sensor**")
        solara.ToggleButtonsSingle(value=sensor, values = sensors)

    if "tuk" in measurement.value:
        m = lk.Tuk(tree.value, measurement.value)
    else:
        m = lk.Measurement(tree.value, measurement.value)
    data = m.sensor_data(sensor.value)
    if len(data) == 0:
        return None

    with solara.lab.Tabs():
        with solara.lab.Tab("Time domain"):
            if data is not None:
                plot(data, resample="A" in sensor.value, title = f"Dataset: {m.tree}, {m.measurement}, {sensor.value}")
                solara.Text(
f"""
Release time is established as a maximum of ext02 for "lano0X" and as zero for "tuk0X".
Here we consider release time {m.release_time}.
""")

        with solara.lab.Tab("Frequency domain (FFT)"):
            with solara.Card():
                fft_image = m.signal(sensor=sensor.value).fft
                plot(fft_image, 
                     resample=False, title = f"FFT for dataset: {m.tree}, {m.measurement}, {sensor.value}",
                     log_y=True, range_x=[0,10],
                     xaxis = "Frequency / Hz"
                     )
            with solara.Card():
                s = m.signal(sensor=sensor.value)
                plot(pd.DataFrame(data = s.tukey, index = s.time), 
                    resample=True, 
                    title = f"Dataset: {m.tree}, {m.measurement}, {sensor.value}")

        with solara.lab.Tab("Frequency domain (Welch)"):
            with solara.Row():
                solara.Markdown(r"$n$ (where $\text{nperseg}=2^n$)")
                solara.ToggleButtonsSingle(values=list(range(6,13)), value=n)
            welch_image = m.signal(sensor=sensor.value).welch(n=n.value)
            plot(welch_image, 
                 resample=False, title = f"Welch spectrum for dataset: {m.tree}, {m.measurement}, {sensor.value}",
                 log_y=True, xaxis = "Frequency / Hz"
                 )
            
            
import solara
import os
from io import BytesIO
import plotly.express as px

from dynatree.pulling import PullingTest, major_minor_axes, slopes

DIRECTORY = '../data/ema'
files = [f.replace(".TXT","") for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f)) and 'TXT' in f]
files.sort()
files_short = [f for f in files if len(f) < 3]

df_majorminor = major_minor_axes()

file = solara.reactive(files[0])
styles_css = """
        .widget-image{width:100%;} 
        .v-btn-toggle{display:inline;}  
        .v-btn {display:inline; text-transform: none;} 
        .vuetify-styles .v-btn-toggle {display:inline;} 
        .v-btn__content { text-transform: none;}
        """

widths = [800,1000,1200, 1400]
width = solara.reactive(1000)
heights = [400,600,800, 1000]
height = solara.reactive(400)

draw_force = solara.reactive(True)
draw_elasto = solara.reactive(False)
draw_inclino_major = solara.reactive(True)
draw_inclino = solara.reactive(False)
use_custom_file = solara.reactive(False)
static_restriction = solara.reactive(True)

@solara.component
def ImageSizes():
    with solara.Card("Image setting"):
        with solara.Column():
            solara.SliderValue(label="width", value=width, values=widths)
            solara.SliderValue(label="height", value=height, values=heights)

content = solara.reactive(None)
external_file = solara.reactive("")

def on_file(f):
    content.value = f["file_obj"].read()
    external_file.value = f["name"]

@solara.component
def Page():
    solara.Style(styles_css)
    with solara.Sidebar():
        solara.Switch(label = "Consider only static data", value = static_restriction)
        if static_restriction.value:
            solara.ToggleButtonsSingle(values=files_short, value=file)
        else:
            solara.ToggleButtonsSingle(values=files, value=file)
        solara.Switch(label = "Draw force", value = draw_force)
        solara.Switch(label = "Draw extensometers", value = draw_elasto)
        solara.Switch(label = "Draw major inclinometers (taken from the table, available only for predefined data)", value = draw_inclino_major)
        solara.Switch(label = "Draw all inclinometers", value = draw_inclino)
        ImageSizes()

        with solara.Card(title = "Input my own file"):
            solara.Switch(label="Use uploaded file", value=use_custom_file)
            solara.FileDrop(
                    label="Drop your own TXT file here. You get images with Force(100), Elasto* and Inclino* data.",
                    on_file=on_file,
            )

    if not use_custom_file.value:
        t = PullingTest(file.value+".TXT", directory=DIRECTORY, localfile=True)
        title = f"Data from {file.value}.TXT"
    else:
        draw_inclino.value = True
        try:
            t = PullingTest(BytesIO(content.value), localfile=False)
            title = f"Data from user file {external_file.value}"
        except:
            solara.Info("Upload the file please.")
            return
    intervals = t.intervals_of_interest()
    with solara.lab.Tabs(lazy=True):
        with solara.lab.Tab("Grafy"):
            grafy(t, intervals,title)
        with solara.lab.Tab("Regrese"):
            regresni_grafy(t,intervals)
        with solara.lab.Tab("Regresní koeficienty"):
            koeficienty()

@solara.component
def koeficienty():
    ansdata = slopes()
    solara.FileDownload(ansdata.to_csv(), filename="regresni_koeficienty.csv", label="Download data as csv")
    solara.display(ansdata.sort_index().style.background_gradient(axis=0))

def grafy(t,intervals,title):

    if draw_force.value:
        solara.Markdown("**Force**")
        fig = t.data.plot(y="Force(100)", backend="plotly")
        for a,b in intervals:
            fig.add_vrect(x0=a, x1=b,
                          fillcolor="gray", opacity=0.2,
                          layer="below", line_width=0)

        fig.update_layout(
            height=height.value,  # Nastavení výšky grafu na 600 pixelů
            width=width.value,  # Nastavení výšky grafu na 600 pixelů
            title = title,
            template="plotly_white"
        )
        solara.FigurePlotly(fig)

    if draw_elasto.value:
        solara.Markdown("**Extensometer**")
        elasto_columns = [col for col in t.data.columns if col.startswith("Elasto")]
        fig = t.data[elasto_columns].plot(backend="plotly")
        for a, b in intervals:
            fig.add_vrect(x0=a, x1=b,
                          fillcolor="gray", opacity=0.2,
                          layer="below", line_width=0)
        fig.update_layout(
            height=height.value,  # Nastavení výšky grafu na 600 pixelů
            width=width.value,  # Nastavení výšky grafu na 600 pixelů
            title=title,
            template="plotly_white"
        )
        solara.FigurePlotly(fig)

    if draw_inclino.value or draw_inclino_major.value:
        solara.Markdown("**Inclinometers**")
        if use_custom_file.value:
            inclino_columns = [col for col in t.data.columns if col.startswith("Inclino")]
        else:
            if draw_inclino.value:
                inclino_columns = [col for col in t.data.columns if col.startswith("Inclino")]
            else:
                inclino_columns = df_majorminor.loc[file.value,:]
        fig = t.data[inclino_columns].plot(backend="plotly")
        for a,b in intervals:
            fig.add_vrect(x0=a, x1=b,
                          fillcolor="gray", opacity=0.2,
                          layer="below", line_width=0)
        fig.update_layout(
            height=height.value,  # Nastavení výšky grafu na 600 pixelů
            width=width.value,  # Nastavení výšky grafu na 600 pixelů
            title=title,
            template="plotly_white")
        solara.FigurePlotly(fig)
    solara.FileDownload(t.data.to_csv(), filename=f"data_{file.value}.csv", label="Download data as csv")

def regresni_grafy(t,intervals):
    try:
        inclinometers = df_majorminor.loc[file.value, :]
        df = t.data.loc[:, ["Force(100)", *inclinometers]].abs()
    except:
        inclinometers = [f for f in t.data.columns if "Inclino" in f]
        df = t.data.loc[:, ["Force(100)", *inclinometers]].abs()
    df = df.interpolate(method='index').dropna()
    for i,[a,b] in enumerate(intervals):
        subdf = df.loc[a:b,:]
        subdf = subdf.set_index("Force(100)")

        fig = px.scatter(data_frame=subdf,  width=width.value, height=height.value,
                         template="plotly_white"
                         )
        solara.FigurePlotly(fig)
        solara.FileDownload(subdf.to_csv(), filename=f"data_{file.value}_pull{i}.csv", label="Download data as csv")



import solara
import os
from io import BytesIO

from numpy.core.defchararray import title
from soupsieve import select

from pulling_tests.pulling import PullingTest


DIRECTORY = '../data/ema'
files = [f.replace(".TXT","") for f in os.listdir(DIRECTORY) if os.path.isfile(os.path.join(DIRECTORY, f))]
files.sort()

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
draw_inclino = solara.reactive(True)
use_custom_file = solara.reactive(False)

@solara.component
def ImageSizes():
    with solara.Card("Image setting"):
        with solara.Column():
            solara.SliderValue(label="width", value=width, values=widths)
            solara.SliderValue(label="height", value=height, values=heights)

content = solara.reactive(None)

def on_file(f):
    content.value = f["file_obj"].read()

@solara.component
def Page():
    solara.Style(styles_css)
    with solara.Sidebar():
        solara.ToggleButtonsSingle(values=files, value=file)
        solara.Switch(label = "Draw force", value = draw_force)
        solara.Switch(label = "Draw extensometers", value = draw_elasto)
        solara.Switch(label = "Draw inclinometers", value = draw_inclino)
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
        try:
            t = PullingTest(BytesIO(content.value), localfile=False)
            title = f"Data from user file"
        except:
            solara.Info("Upload the file please.")
            return

    if draw_force.value:
        solara.Markdown("**Force**")
        fig = t.data.plot(y="Force(100)")
        fig.update_layout(
            height=height.value,  # Nastavení výšky grafu na 600 pixelů
            width=width.value,  # Nastavení výšky grafu na 600 pixelů
            title = title
        )
        solara.FigurePlotly(fig)

    if draw_elasto.value:
        solara.Markdown("**Extensometer**")
        elasto_columns = [col for col in t.data.columns if col.startswith("Elasto")]
        fig = t.data[elasto_columns].plot()
        fig.update_layout(
            height=height.value,  # Nastavení výšky grafu na 600 pixelů
            width=width.value,  # Nastavení výšky grafu na 600 pixelů
            title=title
        )
        solara.FigurePlotly(fig)

    if draw_inclino.value:
        solara.Markdown("**Inclinometers**")
        inclino_columns = [col for col in t.data.columns if col.startswith("Inclino")]
        fig = t.data[inclino_columns].plot()
        fig.update_layout(
            height=height.value,  # Nastavení výšky grafu na 600 pixelů
            width=width.value,  # Nastavení výšky grafu na 600 pixelů
            title=title
        )
        solara.FigurePlotly(fig)


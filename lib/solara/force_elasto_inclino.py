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
  Je proto potřeba spustit  skript `parquet_add_inclino.py` pro začlenění informací do datových souboru,
  dále `extract_release_data.py` pro opravená data před vypuštěním a případně 
  `plot_probes_inclino_force.py` pro obrázky jaké jsou zde.
* Zadáním hodnot si můžeš změnit rozsah na ose x, aby šla dobře vidět kvalita nebo nekvalita
  synchronizace a aby se dalo posoudit, jestli je rozsah před vypuštěním (žlutý pás) umístěn rozumně. (Žlutý pás se zatím /2024-08-20/ k ničemu nepoužívá.) Obě hodnoty nulové znamenají celý rozsah.  

"""

import solara
from solara.lab import task
from plot_probes_inclino_force import plot_one_measurement
from lib_dynatree import timeit
from lib_dynatree import DynatreeMeasurement, find_finetune_synchro
import lib.solara.select_source as s
from static_pull import DynatreeStaticMeasurement
import plotly.graph_objects as go
from plotly.subplots import make_subplots
major_minor = solara.reactive(True)

probes = ["Pt3 with Pt4","Pt3 with fixes"]

probe = solara.reactive(probes[0])
start = solara.reactive(0)
end = solara.reactive(0)

def reset_measurement(a):
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
@timeit
def nakresli():
#    if measurement.value not in available_measurements(df, day.value, tree.value):
#        fig, ax = plt.subplots()
#        return fig
    if probe.value=="Pt3 with Pt4":
        plot_fixes = False
        plot_Pt4 = True
    else:
        plot_fixes = True
        plot_Pt4 = False
    if (end.value == 0) or (end.value <= start.value):
        endlim = None
    else:
        endlim = end.value
    m = DynatreeMeasurement(s.day.value, s.tree.value, s.measurement.value)
    if not nakresli.is_current():
        print("Interrupting non current function nakresli")
        return None

    if not nakresli.is_current():
        print("Interrupting non current function nakresli")
        return None
    try:
        fig = plot_one_measurement(
                date=s.day.value,
                tree=s.tree.value, 
                measurement=s.measurement.value, 
                path=DATA_PATH,
                plot_fixes=probe.value=="Pt3 with fixes", 
                plot_Pt4=probe.value=="Pt3 with Pt4",
                xlim=(start.value,endlim),
                # df_extra=df_ext,
                # df=DF,
                return_figure=True,
                major_minor=major_minor.value
                )    
    except:
        return None
    return fig
    

tab_value = solara.reactive(1)
@solara.component
def Page():
    solara.Style(".widget-image{width:100%;} .v-btn-toggle{display:inline;}  .v-btn {display:inline; text-transform: none;} .vuetify-styles .v-btn-toggle {display:inline;} .v-btn__content { text-transform: none;}")
    solara.Title("DYNATREE: optics, inclinometers, elastometer, force synchro")
    with solara.Sidebar():
        s.Selection(optics_switch = False)
    with solara.lab.Tabs(value=tab_value):
        with solara.lab.Tab("Statický graf"):
            if tab_value.value == 0:
                Sidebar()
                Grafy()
        with solara.lab.Tab("Interaktivní graf"):
            if tab_value.value == 1:
                Interaktivni_grafy()
        with solara.lab.Tab("Návod"):
            solara.Markdown(navod)

delta_time_manual = solara.reactive(0.0)      
use_manual_delta = solara.reactive(False)
@solara.component
def Interaktivni_grafy():
    m = DynatreeStaticMeasurement(
        day=s.day.value, 
        tree=s.tree.value, 
        measurement=s.measurement.value
        )
    with solara.Sidebar():
        s.ImageSizes()
    try:
        df_pull = m.data_pulling
        df_pt = m.data_optics
        df_pt = df_pt - df_pt.iloc[0,:]
    except:
        solara.Error("Měření se nepodařlo zpracovat.")
        if not m.is_optics_available:
            solara.Error("Data pro optiku nejsou k dispozici.")
        return


    delta_time = find_finetune_synchro(m.day, m.tree, m.measurement)
    with solara.Columns(2):
        with solara.Card():
            solara.InputFloat(label="ruční doladění synchronizace", 
                              value=delta_time_manual, )
            solara.Switch(label="Použít pro synchronizaci manuální hodnotu místo předdefinované hodnoty v csv souboru.", value=use_manual_delta)
            solara.Text(f"{s.day.value},{s.tree.value},{s.measurement.value},{delta_time_manual.value},,,,,,,,,,")
        with solara.Column():
            solara.Text(f"'release time' síla (maximum neposunuté síly): {m.release_time_force}")
            solara.Text(f"'release time' optika (maximum Pt3): {m.release_time_optics}")
            solara.Text(f"'delta time' optika (kolonka z csv souboru): {delta_time}")

    if use_manual_delta.value:
        delta_time_final = delta_time_manual.value
    else:
        delta_time_final = delta_time

    fig = make_subplots(rows=2, cols=1,
                    # vertical_spacing = 0.05,
                    shared_xaxes=True,
                   )
    for pt,color in zip(["Pt3","Pt4"], ["blue","red"]):
        fig.add_trace(go.Scatter(x = df_pt.index, 
                             y = df_pt[(pt,"Y0")], 
                             line_color = color,
                             showlegend = True,
                             name = pt,
                            ),
                  row = 1,
                  col = 1,
                  secondary_y = False)
    fig.add_trace(go.Scatter(
                        x = df_pull.index-m.release_time_force
                            +m.release_time_optics+delta_time_final, 
                        y = df_pull["Force(100)"], 
                        line_color = 'blue',
                        showlegend = True,
                        name = "Force",
                        ),
        row = 2,
        col = 1,
        secondary_y = False)
    fig.add_trace(go.Scatter(
                        x = m.data_optics_extra.index, 
                        y = m.data_optics_extra[("Force(100)",'nan')], 
                        line_color = 'green',
                        showlegend = True,
                        name = "Force (předdefinovaná)",
                        ),
        row = 2,
        col = 1,
        secondary_y = False)
    fig.update_layout(height=s.height.value, width=s.width.value,
        hovermode = "x unified",
        legend_traceorder="normal", 
        title = f"Synchronization for {m.measurement_type} {m.day} {m.tree} {m.measurement}"
    )

    fig.update_traces(xaxis='x2') 
    
    solara.FigurePlotly(fig)
    

@solara.component
def Sidebar():
    with solara.Sidebar():
        with solara.Card(title="Image setting"):
            solara.ToggleButtonsSingle(value=probe, values=probes)
            with solara.Tooltip("Plot Major and Minor rather than X and Y. Also rename inclinometers to blue/yellow and make the maximal value positive."):
                solara.Switch(label="Label as Major/Minor", value=major_minor)
            solara.InputFloat("Start", value=start, continuous_update=False)   
            with solara.Tooltip("End of the plot. If 0, then the end of the plot is the end of the data."):             
                solara.InputFloat("End", value=end, continuous_update=False)                
            solara.Div(style={"margin-bottom": "10px"})
            with solara.Row():
                solara.Button("Run calculation", on_click=nakresli, color="primary")
    

@solara.component
def Grafy():
    solara.ProgressLinear(nakresli.pending)
    solara.Warning(solara.Markdown(
"""
* Grafy vznikají za běhu. **Jsou použita předpočítaná data**, 
  aby bylo vidět přesně to, co jde do zpracování. 
* Nezohledňují se případné následné posuny v synchronizaci nebo instrukce pro 
  nulování inklinometrů, které byly zaneseny po posledním spuštění skriptu 
  `parquet_add_inclino.py`. Tato data jdou vidět na sousední záložce 
  (Interaktivní graf). 
* Po změnách v `csv/synchronization_finetune_inclinometers_fix.csv` 
  spusť `parquet_add_inclino.py` a změny se projeví i zde a ve výpočtech.
"""))
    # if measurement.value not in available_measurements(df, day.value, tree.value):
    #     solara.Error(f"Measurement {measurement.value} not available for this tree.")
    #     return
    
    if nakresli.finished:
        if nakresli.value is None:
            solara.Error(solara.Markdown(
"""
* Obrázek nebyl vytvořen. 
* Možná nejsou k dispozici potřebná data (není zpracována optika) nebo došlo k nějaké chybě. 
* Zkontroluj, jestli je zpracována optika a případně nahlaš problém.
"""
))
        else:
            solara.FigureMatplotlib(nakresli.value)
    elif nakresli.not_called:
        with solara.Card(title="Instrukce"):
            solara.Markdown(
                """
                * V bočním panelu vyber měření a stiskni tlačítko Run calculation. 
                * Výpočet je drahý na čas a nespouští se automaticky při změně parametrů. Tlačítko  "Run calculation" použij po každé změně vstupních údajů. 
                * U některých položek je nápověda, která se objeví při najetí myší.
                * Popis co se kreslí a na co se dívat je na kartě Návod.
                """)
    else:
        with solara.Row():
            solara.Text("Pracuji jako ďábel. Může to ale nějakou dobu trvat.")
            solara.SpinnerSolara(size="100px")

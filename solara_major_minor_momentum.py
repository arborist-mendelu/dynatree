#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:00:04 2024

@author: marik
"""

DATA_PATH = "../data"

title = "Pulling, force, inclinometers, elastometer"

import solara
from solara.lab import task
import solara.lab
import solara.express as px
import glob
import pandas as pd
import static_pull
import matplotlib.pyplot as plt
# import hvplot.pandas

def split_path(file):
    data = file.split("/")
    data[-1] = data[-1].replace(".TXT","")
    return [file,data[-2].replace("_","-")] + data[-1].split("_")

def get_all_measurements(cesta=DATA_PATH):
    """
    Get dataframe with all measurements. The dataframe has columns
    date, tree and measurement.
    """
    files = glob.glob(cesta+"/pulling_tests/*/BK_??_M?.TXT")       
    files = [i.replace("BK_","BK") for i in files]
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

day = solara.reactive(days[0])
tree = solara.reactive(trees[0])
measurement = solara.reactive(measurements[0])

xdata = solara.reactive("M_PT")
ydata = solara.reactive(["blue","yellow"])
pull = solara.reactive(0)
restrict_data = solara.reactive(True)
interactive_graph = solara.reactive(False)
ignore_optics_data = solara.reactive(False)
all_data = solara.reactive(False)
force_interval = solara.reactive("None")

def fix_input(a):
    if len(a)==0:
        return ["Force(100)"]
    else:
        return a
    
@task
def nakresli():
    return static_pull.nakresli(day.value, tree.value, measurement.value, ignore_optics_data.value)
    
@solara.component
def Page():
    solara.Title(title)
    solara.Style(".widget-image{width:100%;} .v-btn-toggle{display:inline;}  .v-btn {display:inline; text-transform: none;} .vuetify-styles .v-btn-toggle {display:inline;} .v-btn__content { text-transform: none;}")
    with solara.Sidebar():
        # solara.Markdown("Výběr proměnných pro záložku \"Volba proměnných a regrese\".")
        Selection()
    with solara.lab.Tabs():
        with solara.lab.Tab("Grafy"):
            Graphs()
        with solara.lab.Tab("Volba proměnných a regrese"):
            try:
                Detail()
            except:
                solara.Info("Nejprve záložka Grafy")
        with solara.lab.Tab("Návod a komentáře"):
            Help()        
        
    # MainPage()

# def clear():
#     # https://stackoverflow.com/questions/37653784/how-do-i-use-cache-clear-on-python-functools-lru-cache
#     static_pull.nakresli.__dict__["__wrapped__"].cache_clear()
#     static_pull.proces_data.__dict__["__wrapped__"].cache_clear()
    

def Selection():
    with solara.Card():
        with solara.Column():
            solara.ToggleButtonsSingle(value=day, values=list(days), on_value=lambda x: measurement.set(measurements[0]))
            solara.ToggleButtonsSingle(value=tree, values=list(trees), on_value=lambda x: measurement.set(measurements[0]))
            with solara.Row():
                solara.ToggleButtonsSingle(value=measurement, 
                                           values=available_measurements(df, day.value, tree.value),
                                           on_value=lambda x:nakresli()
                                           )
                with solara.Tooltip("Umožní ignorovat preprocessing udělaný na tahovkách M2 a více. Tím bude stejná metodika jako pro M01 (tam se preprocessing nedělal), ale přijdeme o časovou synchronizaci s optikou a hlavně přijdeme o opravu vynulování inklinometrů."):
                    solara.Switch(label="Ignore prepocessed files for M2 and higher", value=ignore_optics_data)
        solara.Div(style={"margin-bottom": "10px"})
        with solara.Row():
            solara.Button("Run calculation", on_click=nakresli, color="primary")
            # solara.Button("Clear cache", on_click=clear(), color="primary")
            solara.Markdown(f"**Selected**: {day.value}, {tree.value}, {measurement.value}")   
            
def Graphs():            
    solara.ProgressLinear(nakresli.pending)    
    if measurement.value not in available_measurements(df, day.value, tree.value):
        solara.Error(f"Measurement {measurement.value} not available for this tree.")
        return

    if nakresli.not_called:
        solara.Info("Vyber měření a stiskni tlačítko \"Run calculation\". Ovládací prvky jsou v sidebaru. Pokud není otevřený, otevři klilnutím na tři čárky nalevo v modrém pásu.")
        solara.Warning("Pokud pracuješ v prostředí JupyterHub, asi bude lepší aplikaci maximalizovat. Tlačítko je v modrém pásu úplně napravo.")
    elif not nakresli.finished:
        with solara.Row():
            solara.Text("Pracuji jako ďábel. Může to ale nějakou dobu trvat.")
            solara.SpinnerSolara(size="100px")
    else:
        solara.Markdown("Na obrázku je průběh experimentu (časový průběh síly) a detaily pro rozmezí 10%-90% maxima síly. \n\n V detailech je časový průběh síly, časový průběh na inklinometrech a grafy inklinometry versus síla nebo moment. Jestli moment z Rope nebo z tabulky PT viz karta s Návod a komentáře.")
        f = nakresli.value
        with solara.ColumnsResponsive(6): 
            for _ in f:
                solara.FigureMatplotlib(_)
        plt.close('all')
        # data['dataframe']["Time"] = data['dataframe'].index
        # solara.DataFrame(data['dataframe'], items_per_page=20)
        # cols = data['dataframe'].columns
def Detail():
    data = static_pull.proces_data(day.value, tree.value, measurement.value, ignore_optics_data.value)
    if nakresli.not_called:
        solara.Info("Nejdřív nakresli graf v první záložce.")        
    elif not nakresli.finished:
        with solara.Row():
            solara.Text("Pracuji jako ďábel. Může to ale nějakou dobu trvat.")
            solara.SpinnerSolara(size="100px")
    else:
        with solara.Card():
            solara.Markdown("## Increasing part of the time-force diagram")
            solara.Markdown("""
                    Pro výběr proměnných na vodorovnou a svislou osu otevři menu v sidebaru (tři čárky v horním panelu). Po výběru můžeš sidebar zavřít. Přednastavený je moment vypočítaný z geometrie na vodorovné ose a oba inlinometry na svislé ose.
                    """)
            
            with solara.Sidebar():
                cols = ['Time','Force(100)', 'Elasto(90)', 'Elasto-strain',
                        # 'Inclino(80)X', 'Inclino(80)Y', 'Inclino(81)X', 'Inclino(81)Y', 
                'RopeAngle(100)',
                'blue', 'yellow', 'blueX', 'blueY', 'yellowX', 'yellowY', 
                'blue_Maj', 'blue_Min', 'yellow_Maj', 'yellow_Min',
                'F_horizontal_Rope', 'F_vertical_Rope',
                'M_Rope', 'M_Pt_Rope', 'F_horizontal_PT', 'F_vertical_PT', 'M_PT',
                'M_Pt_PT']
                with solara.Card():
                    solara.Markdown("### Horizontal axis \n\n Choose one variable.")
                    solara.ToggleButtonsSingle(values=cols, value=xdata, dense=True)
                with solara.Card():
                    solara.Markdown("### Vertical axis \n\n Choose one or more variables.")
                    solara.ToggleButtonsMultiple(values=cols, value=ydata, dense=True)
                pulls = list(range(len(data['times'])))
            
            if measurement.value == "M1":
                with solara.Row():
                    solara.Markdown("Pull No. of M1:")
                    solara.ToggleButtonsSingle(values=pulls, value=pull)
                pull_value = pull.value
            else:
                pull_value = 0
            subdf = data['dataframe'].loc[data['times'][pull_value]['minimum']:data['times'][pull_value]['maximum'],:]

            with solara.Row():
                solara.Switch(label="Restrict 10%-90% of Fmax", value=restrict_data)
                with solara.Tooltip("Umožní zobrazit graf pomocí plotly. Bude možné zoomovat, odečítat hodnoty, klikáním na legendu skrývat a odkrývat proměnné apod."):
                    solara.Switch(label="Interactive graph", value=interactive_graph)
                if restrict_data.value:
                    lower, upper = static_pull.get_interval_of_interest(subdf)        
                    subdf = subdf.loc[lower:upper,:]            
                with solara.Tooltip("Umožní zobrazit grafy veličin pro celý časový průběh."):
                    solara.Switch(label="Ignore time restriction", value=all_data)
                if all_data.value:
                    subdf = data['dataframe']

            title = f"{day.value} {tree.value} {measurement.value} Pull {pull.value}"
            if interactive_graph.value:
                kwds = {"template":"plotly_white", "height":600, "title":title}
                # kwds = {"height":600, "title":title}
                try:
                    if xdata.value == "Time":
                        px.scatter(subdf.astype(float), y=fix_input(ydata.value), **kwds)
                    else:
                        px.scatter(subdf.astype(float), x=xdata.value, y=fix_input(ydata.value), **kwds)
                except:
                    solara.Error(solara.Markdown("""### Image failed. 
                                 
* Something is wrong. Switch to noninteractive plot or change variables setting. 
* This error appears especially if you try to plot both forces and inclinometers on vertical axis.
"""
                                 ))
            else:
                fig, ax = plt.subplots()
                if xdata.value == "Time":
                    subdf["Time"] = subdf.index
                subdf.plot(x=xdata.value, y=ydata.value, style='.', ax=ax)
                ax.grid()
                ax.set(title = title)
                solara.FigureMatplotlib(fig)
            
            try:
                # find regresions
                if xdata.value != "Time":
                    subsubdf = subdf.loc[:,[xdata.value]+[i for i in ydata.value if i!=xdata.value]]
                    reg_df = static_pull.get_regressions(subsubdf, xdata.value)
                    solara.DataFrame(reg_df.iloc[:,:4])                
            except:
                solara.Error("Něco se pokazilo při hledání regresí. Nahlaš prosím problém. Pro další práci vyber jiné veličiny.")


def Help():
    solara.Markdown(
"""
## Návod

### Práce

* **Pokud máš otevřeno v zápisníku Jupyteru, bude vhodnější maximalizovat aplikaci. V modrém pásu napravo je ikonka na maximalizaci uvnitř okna prohlížeče.**
* Vyber datum, strom a měření. Pokud se obrázek neaktualizuje automaticky, klikni na tlačítko pro spuštění výpočtu. Výpočet se spustí kliknutím tlačítka nebo změnou volby měření. Pokud se mění strom nebo den a měření zůstává M1, je potřeba stisknout tlačítko.
* Zobrazí se průběh experimentu, náběh (resp. tři náběhy) síly do maxima a zvýrazněná část pro analýzu. Ta je 10-90 procent, ale možná má smyslu zkusit i jiné meze.
* Poté máš možnost si detailněji vybrat, co má být v dalším grafu na vodorovné a svislé ose. Tlačítka pro výběr se objeví v bočním panelu, aby se dala skrývat a nezavazela. Počítá se regrese mezi veličinou na vodorovné ose a každou z veličin na ose svislé.

### Popis

* Inlinometr blue je 80, yelllow je 81. Výchylky v jednotlivých osách jsou blueX a blueY resp. blue_Maj a blue_Min. Celková výchylka je blue. Podobně  druhý inklinometr.
* F se rozkládá na vodorovnou a svislou složku.Vodorovná se používá k výpočtu momentu v bodě úvazu (M) a v bodě Pt3 (M_PT). K tomu je potřeba znát odchylku lana od vodorovné polohy. Toto je možné 
    1. zjistit ze siloměru v pull TXT souborech jako Ropeangle(100) 
    2. anebo použít fixní hodnotu z geometrie a měření délek. 
* Druhá varianta je spolehlivější, **Rope(100) má někdy celkem rozeskákané hodnoty.**
  Vypočítané veličiny mají na konci _Rope nebo _PT. Asi se hodí více ty s _PT na konci.
* Elasto-strain je Elasto(90)/200000.

### Komenáře

* V diagramech síla nebo moment versus inklinometry není moc změna trendu mezi první polovinou diagramu a celkem. Takže je asi jedno jestli bereme pro sílu rozmezí 10-90 procent Fmax nebo 10-40 procent.
* Veličina Rope(100) ze siloměru má dost rozeskákané hodnoty a to zašpiní cokoliv, co se pomocí toho počítá. Asi nebrat. To jsou veličiny, které mají na konci text "_Rope". Místo nich použít ty, co mají na konci "_PT"
* Graf moment versus inlinometry má někdy na začátku trochu neplechu. Možná mají velký vliv nevynulované hodnoty 
  inklinometrů, protože se přidávají k malým náklonům a hodně zkreslují. Zvážit posunutí rozmezí na vyšší hodnotu než 10 procent Fmax.

### Data

Je rozdíl mezi daty ze statiky a pull-release.
Data pro M01 jsou přímo z TXT souborů produkovaných přístrojem. Data pro další 
měření (M02 a výše) byla zpracována: 
    
* počátek se sesynchronizoval s optikou, 
* data se interpolovala na stejné časy jako v optice (tedy je více dat) 
* a někdy se ručně opravilo nevynulování nebo nedokonalé vynulování inklinoměru. 

Dá se snadno přepnout na to, aby se všechna data brala z TXT souborů (volba `skip_optics` ve funkci `get_static_pulling_data`), ale přišli bychom o opravy s vynulováním. Resp. bylo by potřeba to zapracovat.



"""
        )
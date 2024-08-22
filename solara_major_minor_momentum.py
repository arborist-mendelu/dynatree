#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:00:04 2024

@author: marik
"""

DATA_PATH = "../data"

title = "DYNATREE: pulling, force, inclinometers, extensometer, optics, ..."

import solara
from solara.lab import task
import solara.lab
import solara.express as px
import glob
import pandas as pd
import static_pull
import matplotlib.pyplot as plt
import numpy as np
import lib_dynatree
from static_pull import get_all_measurements, available_measurements


df = get_all_measurements()
days = df["day"].drop_duplicates().values
trees = df["tree"].drop_duplicates().values
measurements = df["measurement"].drop_duplicates().values

day = solara.reactive(days[0])
tree = solara.reactive(trees[0])
measurement = solara.reactive(measurements[0])

xdata = solara.reactive("M_Measure")
ydata = solara.reactive(["blue","yellow"])
ydata2 = solara.reactive([])
pull = solara.reactive(0)
restrict_data = solara.reactive(True)
interactive_graph = solara.reactive(False)
ignore_optics_data = solara.reactive(False)
all_data = solara.reactive(False)
force_interval = solara.reactive("None")

def fix_input(a):
    """
    The input is a list. 
    
    The output is ["Force(100)"] is the input is empty. If not empty, the 
    input is copied to output.
    
    Used for drawing. If no value is selected for the graph, the Force(100) 
    is plotted.
    """
    if len(a)==0:
        return ["Force(100)"]
    return a
    
@task
def nakresli(reset_measurements = False):
    if reset_measurements:
        measurement.set(measurements[0])
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
                solara.Error("Něco se nepovedlo. Možná není vybráno nic pro svislou osu. nebo je vybrána stejná veličina pro vodorovnou a svislou osu. Nebo je nějaký jiný problém. Možná mrkni nejprve na záložku Grafy")
                # lib_dynatree.logger.error("Solara: Kreslení grafu selhalo")
        with solara.lab.Tab("Návod a komentáře"):
            Help()        
        
    # MainPage()

# def clear():
#     # https://stackoverflow.com/questions/37653784/how-do-i-use-cache-clear-on-python-functools-lru-cache
#     static_pull.nakresli.__dict__["__wrapped__"].cache_clear()
#     static_pull.process_data.__dict__["__wrapped__"].cache_clear()

def Selection():
    with solara.Card():
        with solara.Column():
            solara.ToggleButtonsSingle(value=day, values=list(days), on_value=lambda x: nakresli(reset_measurements=True))
            solara.ToggleButtonsSingle(value=tree, values=list(trees), on_value=lambda x: nakresli(reset_measurements=True))
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
        solara.Info(solara.Markdown("""
                        
* Vyber měření a stiskni tlačítko \"Run calculation\". 
* Ovládací prvky jsou v sidebaru. Pokud není otevřený, otevři kliknutím na tři čárky nalevo v modrém pásu.
* Při změně vstupů se většinou obrázek aktualizuje, ale ne vždy. Pokud nadpis na obrázku nesouhlasí s vybranými hodnotami, spusť výpočet tlačítkem \"Run calculation\".

                                    """))
        solara.Warning("Pokud pracuješ v prostředí JupyterHub, asi bude lepší aplikaci maximalizovat. Tlačítko je v modrém pásu úplně napravo.")
    elif not nakresli.finished:
        with solara.Row():
            solara.Text("Pracuji jako ďábel. Může to ale nějakou dobu trvat.")
            solara.SpinnerSolara(size="100px")
    else:
        solara.Markdown("Na obrázku je průběh experimentu (časový průběh síly) a detaily pro rozmezí 10%-90% maxima síly. \n\n V detailech je časový průběh síly, časový průběh na inklinometrech a grafy inklinometry versus síla nebo moment. Jestli moment z Rope nebo z tabulky PT viz karta Návod a komentáře.")
        f = nakresli.value
        with solara.ColumnsResponsive(6): 
            for _ in f:
                solara.FigureMatplotlib(_)
        plt.close('all')
        # data['dataframe']["Time"] = data['dataframe'].index
        # solara.DataFrame(data['dataframe'], items_per_page=20)
        # cols = data['dataframe'].columns
def Detail():
    data = static_pull.process_data(day.value, tree.value, measurement.value, ignore_optics_data.value)
    if nakresli.not_called:
        solara.Info("Nejdřív nakresli graf v první záložce. Klikni na Run calculation v sidebaru.")        
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
                cols = ['Time','Pt3','Pt4','Force(100)', 'Elasto(90)', 'Elasto-strain',
                        # 'Inclino(80)X', 'Inclino(80)Y', 'Inclino(81)X', 'Inclino(81)Y', 
                'RopeAngle(100)',
                'blue', 'yellow', 'blueX', 'blueY', 'yellowX', 'yellowY', 
                'blue_Maj', 'blue_Min', 'yellow_Maj', 'yellow_Min',
                'F_horizontal_Rope', 'F_vertical_Rope',
                'M_Rope', 'M_Pt_Rope', 'M_Elasto_Rope',
                'F_horizontal_Measure', 'F_vertical_Measure',
                'M_Measure', 'M_Pt_Measure', 'M_Elasto_Measure',]
                with solara.Card():
                    solara.Markdown("### Horizontal axis \n\n Choose one variable.")
                    solara.ToggleButtonsSingle(values=cols, value=xdata, dense=True)
                with solara.Card():
                    solara.Markdown("### Vertical axis \n\n Choose one or more variables.")
                    solara.ToggleButtonsMultiple(values=cols[1:], value=ydata, dense=True)
                with solara.Card():
                    solara.Markdown("### Second vertical axis \n\n Choose one variable for right vertical axis. Does not work great in interactive plots. In interactive plots we plot rescaled data. The scale factor is determined from maxima.")
                    solara.ToggleButtonsSingle(values=[None]+cols[1:], value=ydata2, dense=True)
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

            title = f"{day.value} {tree.value} {measurement.value} Pull {pull_value}"
            if interactive_graph.value:
                kwds = {"template":"plotly_white", "height":600, "title":title}
                # kwds = {"height":600, "title":title}
                try:
                    subdf = subdf.astype(float)
                    if ydata2.value!=None: # Try to add rescaled column
                        maximum_target = np.nanmax(subdf[fix_input(ydata.value)].values)
                        maximum_ori = np.nanmax(subdf[ydata2.value].values)
                        print (f"maxima jsou {maximum_target} a {maximum_ori}")
                        subdf.loc[:,f"{ydata2.value}_rescaled"] = subdf.loc[:,ydata2.value]/np.abs(maximum_ori/maximum_target)
                        extradata = [f"{ydata2.value}_rescaled"]
                    else:
                        extradata = []
                    cols_to_draw = fix_input(ydata.value + extradata)
                    if xdata.value == "Time":
                        px.scatter(subdf, y=cols_to_draw, **kwds)
                    else:
                        px.scatter(subdf, x=xdata.value, y=cols_to_draw, **kwds)
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
                if ydata2.value!=None:
                    ax2 = ax.twinx()
                    # see https://stackoverflow.com/questions/24280180/matplotlib-colororder-and-twinx
                    ax2._get_lines = ax._get_lines
                    subdf.plot(x=xdata.value, y=ydata2.value, style='.', ax=ax2)
                    # ask matplotlib for the plotted objects and their labels
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.get_legend().remove()
                    ax2.legend(lines + lines2, labels + labels2)                    
                ax.grid()
                ax.set(title = title)
                solara.FigureMatplotlib(fig)
                
            try:
                # find regresions
                if xdata.value != "Time":
                    # subsubdf = subdf.loc[:,[xdata.value]+[i for i in ydata.value+[ydata2.value] if i!=xdata.value]]
                    if ydata2.value is None:
                        target = ydata.value
                    else:
                        target = [ydata2.value] + ydata.value
                    reg_df = static_pull.get_regressions(
                        subdf,
                        [[xdata.value] + target]
                        )
                    # subdf.to_csv("temp.csv")
                    solara.DataFrame(reg_df.iloc[:,:5])                
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
* F se rozkládá na vodorovnou a svislou složku.Vodorovná se používá k výpočtu momentu v bodě úvazu (M) a v bodě Pt3 (M_Pt). K tomu je potřeba znát odchylku lana od vodorovné polohy. Toto je možné 
    1. zjistit ze siloměru v pull TXT souborech jako Ropeangle(100) 
    2. anebo použít fixní hodnotu z geometrie a měření délek
    3. anebo použít fixní hodnotu naměřenou na začátku experimentu.
* Druhé dvě varianty jsou spolehlivější, **Rope(100) má někdy celkem rozeskákané hodnoty.**  Vypočítané veličiny mají na konci _Rope (varianta 1) nebo _Measure (varianta 3). Varianta 2 bude stejná jsko 3, jenom se lišit konstantním faktorem. Asi se hodí více data s _Measure na konci.
* Elasto-strain je Elasto(90)/200000.

### Komenáře

* V diagramech síla nebo moment versus inklinometry není moc změna trendu mezi první polovinou diagramu a celkem. Takže je asi jedno jestli bereme pro sílu rozmezí 10-90 procent Fmax nebo 10-40 procent.
* Veličina Rope(100) ze siloměru má dost rozeskákané hodnoty a to zašpiní cokoliv, co se pomocí toho počítá. Asi nebrat. To jsou veličiny, které mají na konci text "_Rope". Místo nich použít ty, co mají na konci "_Measure"
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

### Poznámky

|Měření   |Poznámka   |
|:--|:--|
|2021-03-22 BK08 M05| Siloměr neměřil. Není síla ani momenty.|
|2022-08-16 BK13 M02| Optika totálně selhala. TODO: brát jako statiku, viz níže.|
|2022-08-16 BK16 M01| Po zatáhnutí zůstávala velká deformace. Ale zpracování OK.|
|2022-04-05 BK21 M05| Vůbec není v optice. Zatím vyhozeno. TODO: brát jako statiku, viz níže.|

Pokud chceš dynamické měření brát jako statiku, přepni přepínač "Ignore prepocessed files for M2 and higher" (vedle sezamu dostupných měření.)

"""
        )
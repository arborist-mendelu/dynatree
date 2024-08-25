#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:00:04 2024

@author: marik
"""

DATA_PATH = "../data"

tightcols = {'gap':"0px"}
regression_settings = {'color':'gray', 'alpha':0.5}

title = "DYNATREE: pulling, force, inclinometers, extensometer, optics, ..."

import solara
from solara.lab import task
import solara.lab
import solara.express as px
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lib_dynatree
import static_pull
from static_pull import get_all_measurements, available_measurements


df = get_all_measurements(method='all')
days = df["date"].drop_duplicates().values
trees = df["tree"].drop_duplicates().values
measurements = df["measurement"].drop_duplicates().values

day = solara.reactive(days[0])
tree = solara.reactive(trees[0])
measurement = solara.reactive(measurements[0])


data_possible_restrictions = ["0-100%","10%-90%","30%-90%"]

xdata = solara.reactive("M_Measure")
ydata = solara.reactive(["blue","yellow"])
ydata2 = solara.reactive([])
pull = solara.reactive(0)
restrict_data = solara.reactive(data_possible_restrictions[-1])
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
    data_object = lib_dynatree.Dynatree_Measurement(day.value, tree.value, measurement.value,)
    return static_pull.nakresli(data_object, skip_optics=True)
    

@solara.component
def Page():
    solara.Title(title)
    solara.Style(".widget-image{width:100%;} .v-btn-toggle{display:inline;}  .v-btn {display:inline; text-transform: none;} .vuetify-styles .v-btn-toggle {display:inline;} .v-btn__content { text-transform: none;}")
    with solara.Sidebar():
        # solara.Markdown("Výběr proměnných pro záložku \"Volba proměnných a regrese\".")
        Selection()
    with solara.lab.Tabs():
        with solara.lab.Tab("Grafy"):
            with solara.Card():
                Graphs()
        with solara.lab.Tab("Volba proměnných a regrese"):
            with solara.Card():
                Detail()
                     
        with solara.lab.Tab("Návod a komentáře"):
            with solara.Card():
                Help()        
        
    # MainPage()

# def clear():
#     # https://stackoverflow.com/questions/37653784/how-do-i-use-cache-clear-on-python-functools-lru-cache
#     static_pull.nakresli.__dict__["__wrapped__"].cache_clear()
#     static_pull.process_data.__dict__["__wrapped__"].cache_clear()

def Selection():
    data_obj = lib_dynatree.Dynatree_Measurement(day.value, tree.value, measurement.value,)
    with solara.Card():
        with solara.Column():
            solara.ToggleButtonsSingle(value=day, values=list(days), on_value=lambda x: nakresli(reset_measurements=True))
            solara.ToggleButtonsSingle(value=tree, values=list(trees), on_value=lambda x: nakresli(reset_measurements=True))
            with solara.Row():
                solara.ToggleButtonsSingle(value=measurement, 
                                           values=available_measurements(df, day.value, tree.value),
                                           on_value=lambda x:nakresli()
                                           )
                with solara.Tooltip("Umožní ignorovat preprocessing udělaný na tahovkách M02 a více. Tím bude stejná metodika jako pro M01 (tam se preprocessing nedělal), ale přijdeme o časovou synchronizaci s optikou a hlavně přijdeme o opravu vynulování inklinometrů."):
                    solara.Switch(
                        label="Ignore prepocessed files for M2 and higher",
                        value=ignore_optics_data,
                        disabled = not data_obj.is_optics_available
                        )
        solara.Div(style={"margin-bottom": "10px"})
        with solara.Row():
            solara.Button("Run calculation", on_click=nakresli, color="primary")
            # solara.Button("Clear cache", on_click=clear(), color="primary")
            solara.Markdown(f"**Selected**: {day.value}, {tree.value}, {measurement.value}")
        if data_obj.is_optics_available:
            solara.Markdown(f"Optics is available for this measurement.")
        else:            
            solara.Markdown(f"Optics is **not** available for this measurement.")
        
            
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
        solara.Markdown("Na obrázku je průběh experimentu (časový průběh síly) a detaily pro rozmezí 30%-90% maxima síly. \n\n V detailech je časový průběh síly, časový průběh na inklinometrech a grafy inklinometry versus síla nebo moment. Jestli moment z Rope nebo z tabulky PT viz karta Návod a komentáře.")
        f = nakresli.value
        with solara.ColumnsResponsive(6): 
            for _ in f:
                solara.FigureMatplotlib(_)
        plt.close('all')
        solara.Markdown("""
        * Data jsou ze souboru z tahovek. Sample rate cca 0.1 sec. Obrázky jsou jenom pro orientaci a pro kontrolu ořezu dat. Lepší detaily se dají zobrazit na vedlejší kartě s volbou proměnných a regresí.
        * Pokud nevyšla detekce části našeho zájmu, zadej ručně meze, ve kterých hledat. Jsou v souboru `csv/intervals_split_M01.csv` (podadresář souboru se skripty). Potom nahrát na github a zpropagovat do všech zrcadel.
        """
        )        
        # data['dataframe']["Time"] = data['dataframe'].index
        # solara.DataFrame(data['dataframe'], items_per_page=20)
        # cols = data['dataframe'].columns

msg="""
### Něco se nepovedlo. 
                     
* Možná není vybráno nic pro svislou osu. 
* Možná je vybrána stejná veličina pro vodorovnou a svislou osu. 
* Nebo je nějaký jiný problém. Možná mrkni nejprve na záložku Grafy."""

def Detail():
    data_obj = lib_dynatree.Dynatree_Measurement(day.value, tree.value, measurement.value,)
    data = static_pull.process_data(data_obj, ignore_optics_data.value)
    if nakresli.not_called:
        solara.Info("Nejdřív nakresli graf v první záložce. Klikni na Run calculation v sidebaru.")        
        return
    if not nakresli.finished:
        with solara.Row():
            solara.Text("Pracuji jako ďábel. Může to ale nějakou dobu trvat.")
            solara.SpinnerSolara(size="100px")
            return
    # with solara.Card():
    solara.Markdown("## Increasing part of the time-force diagram")
    solara.Markdown("""
            Pro výběr proměnných na vodorovnou a svislou osu otevři menu v sidebaru (tři čárky v horním panelu). Po výběru můžeš sidebar zavřít. Přednastavený je moment vypočítaný z pevného naměřeného úhlu lana na vodorovné ose a oba inlinometry na svislé ose.
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
            solara.Markdown("### Vertical axis \n\n Choose one or more variables. You cannot choose the same variable which has been used for horizontal axis.")
            solara.ToggleButtonsMultiple(values=cols[1:], value=ydata, dense=True)
        with solara.Card():
            with solara.VBox():
                with solara.Tooltip("Choose one variable for second vertical axis, shown on the right. (Only limited support in interactive plots. In interactive plots we plot rescaled data. The scale factor is determined from maxima.) You cannot choose the variable used for horizontal axis."):
                    with solara.VBox():
                        solara.Markdown("### Second vertical axis")
                        solara.Text("(hover here for description)")
            
                solara.ToggleButtonsSingle(values=[None]+cols[1:], value=ydata2, dense=True)
        pulls = list(range(len(data['times'])))

    with solara.Row():
        if measurement.value == "M01":
            with solara.Card():
                with solara.Column(**tightcols):
                    solara.Markdown("Pull No. of M01:")
                    solara.ToggleButtonsSingle(values=pulls, value=pull)
                pull_value = pull.value
        else:
            pull_value = 0
        subdf = data['dataframe'].loc[data['times'][pull_value]['minimum']:data['times'][pull_value]['maximum'],:]

        with solara.Card():
            with solara.Column(**tightcols):
                solara.Markdown("Bounds to cut out boundaries in % of Fmax")
                solara.ToggleButtonsSingle(values=data_possible_restrictions, value=restrict_data)
        with solara.Card():
            with solara.Column(**tightcols):
                with solara.Tooltip("Umožní zobrazit graf pomocí knihovny Plotly. Bude možné zoomovat, odečítat hodnoty, klikáním na legendu skrývat a odkrývat proměnné apod. Nebudou zobrazeny regresní prímky."):
                    solara.Switch(label="Interactive graph", value=interactive_graph)
                with solara.Tooltip("Umožní zobrazit grafy veličin pro celý časový průběh."):
                    solara.Switch(label="Ignore time restriction", value=all_data)
        if restrict_data.value!=data_possible_restrictions[0]:
            up = .90
            if restrict_data.value == data_possible_restrictions[1]:
                lb = .10
            else:
                lb = .30
            lower, upper = static_pull.get_interval_of_interest(subdf, maximal_fraction=up, minimal_fraction=lb)        
            subdf = subdf.loc[lower:upper,:]            
        if all_data.value:
            subdf = data['dataframe']

    try:
        # find regresions
        if xdata.value != "Time":
            # subsubdf = subdf.loc[:,[xdata.value]+[i for i in ydata.value+[ydata2.value] if i!=xdata.value]]
            ydata.value = [i for i in ydata.value if i!=xdata.value]
            if xdata.value == ydata2.value:
                ydata2.value = None
            if ydata2.value is None:
                target = ydata.value
            else:
                target = ydata.value + [ydata2.value]
            reg_df = static_pull.get_regressions(
                subdf,
                [[xdata.value] + target]
                )
            solara.DataFrame(reg_df.iloc[:,:5])                
    except:
        solara.Error("Něco se pokazilo při hledání regresí. Nahlaš prosím problém. Pro další práci vyber jiné veličiny.")

    title = f"{day.value} {tree.value} {measurement.value} Pull {pull_value}"
    if interactive_graph.value:
        kwds = {"template":"plotly_white", "height":600, "title":title}
        # kwds = {"height":600, "title":title}
        try:
            if ydata2.value!=None: # Try to add rescaled column
                maximum_target = np.nanmax(subdf[fix_input(ydata.value)].values)
                maximum_ori = np.nanmax(subdf[ydata2.value].values)
                # print (f"maxima jsou {maximum_target} a {maximum_ori}")
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
        # print(xdata.value, ydata.value)
        try:
            subdf.plot(x=xdata.value, y=ydata.value, style='.', ax=ax)
            if xdata.value != "Time":
                t = np.linspace(*ax.get_xlim(),5)
                for y in ydata.value:
                    if y not in reg_df["Dependent"]:
                        continue
                    d = reg_df[reg_df["Dependent"]==y].loc[:,["Slope","Intercept"]]
                    # print(f"y:{y}\nvalues:{t*d.iat[0,0]+d.iat[0,1]}")
                    ax.plot(t,t*d.iat[0,0]+d.iat[0,1], **regression_settings)
        except:
            # print("Neco je blbe")
            # print(reg_df, ydata.value)                    
            return
            pass
        if ydata2.value!=None:
            ax2 = ax.twinx()
            # see https://stackoverflow.com/questions/24280180/matplotlib-colororder-and-twinx
            ax2._get_lines = ax._get_lines
            subdf.plot(x=xdata.value, y=ydata2.value, style='.', ax=ax2)
            # d = reg_df[reg_df["Dependent"]==ydata2.value].loc[:,["Slope","Intercept"]]
            # ax2.plot(t,t*d.iat[0,0]+d.iat[0,1], **regression_settings)
            # ask matplotlib for the plotted objects and their labels
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.get_legend().remove()
            ax2.legend(lines + lines2, labels + labels2)                    
        ax.grid()
        ax.set(title = title)
        solara.FigureMatplotlib(fig)
                


def Help():
    solara.Markdown(
"""
## Návod

### Práce

* **Pokud máš otevřeno v zápisníku Jupyteru, bude vhodnější maximalizovat aplikaci. V modrém pásu napravo je ikonka na maximalizaci uvnitř okna prohlížeče.**
* Vyber datum, strom a měření. Pokud se obrázek neaktualizuje automaticky, klikni na tlačítko pro spuštění výpočtu. Výpočet se spustí kliknutím tlačítka nebo změnou volby měření. Pokud se mění strom nebo den a měření zůstává M01, je potřeba stisknout tlačítko.
* Zobrazí se průběh experimentu, náběh (resp. tři náběhy) síly do maxima a zvýrazněná část pro analýzu. Ta je 30-90 procent, ale dá se nastavit i 10-90 procent nebo 0-100 procent.
* Poté máš možnost si detailněji vybrat, co má být v dalším grafu na vodorovné a svislé ose. Tlačítka pro výběr se objeví v bočním panelu, aby se dala skrývat a nezavazela. Počítá se regrese mezi veličinou na vodorovné ose a každou z veličin na ose svislé. Regrese nejsou dostupné, pokud je vodorovně čas (nedávalo by smysl) a pokus je na vodorovné a svislé ose stejná veličina (taky by nedávalo smysl).

### Popis

* Inlinometr blue je 80, yelllow je 81. Výchylky v jednotlivých osách jsou blueX a blueY resp. blue_Maj a blue_Min. Celková výchylka je blue. Podobně  druhý inklinometr.
* F se rozkládá na vodorovnou a svislou složku.Vodorovná se používá k výpočtu momentu v bodě úvazu (M), v bodě Pt3 (M_Pt) a v místě s extenzometrem (M_Elasto). K tomu je potřeba znát odchylku lana od vodorovné polohy. Toto je možné 
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

### Historie

* 2024-08-??: první verze
* 2024-08-23: je možné volit ořez dat v procentech Fmax mezi 0-100%, 10%-90% a 30%-90%, zobrazuje se regresní přímka. TODO: najít v datech, kde se to nejvíce liši a nechat info zde.

"""
        )

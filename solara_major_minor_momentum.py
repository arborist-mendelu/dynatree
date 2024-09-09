#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:00:04 2024

@author: marik
"""

from lib_find_measurements import get_all_measurements, available_measurements
import static_pull
import lib_dynatree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import glob
import solara.express as px
import solara.lab
from solara.lab import task
import solara
import time
DATA_PATH = "../data"

tightcols = {'gap': "0px"}
regression_settings = {'color': 'gray', 'alpha': 0.5}

title = "DYNATREE: pulling, force, inclinometers, extensometer, optics, ..."

methods = solara.reactive(['normal', 'den', 'noc', 'afterro', 'afterro2', 'mraz'])
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

data_possible_restrictions = ["0-100%", "10%-90%", "30%-90%"]

xdata = solara.reactive("M_Measure")
ydata = solara.reactive(["blue", "yellow"])
ydata2 = solara.reactive([])
pull = solara.reactive(0)
restrict_data = solara.reactive(data_possible_restrictions[-1])
interactive_graph = solara.reactive(False)
all_data = solara.reactive(False)
force_interval = solara.reactive("None")
# data_from_url = solara.reactive(False)
# 
def fix_input(a):
    """
    The input is a list. 

    The output is ["Force(100)"] is the input is empty. If not empty, the 
    input is copied to output.

    Used for drawing. If no value is selected for the graph, the Force(100) 
    is plotted.
    """
    if len(a) == 0:
        return ["Force(100)"]
    return a

def resetuj_a_nakresli(reset_measurements=False):
    measurement.set(measurements.value[0])
    return nakresli()

# def get_data_object(day, tree, measuemrent, measurement_type, optics):
def get_data_object():
    """
    Return the measurements. 
    If optics is not available, force optics to False.
    
    Cheap, 0.0002s.
    """
    data_object = static_pull.DynatreeStaticMeasurement(
        day=day.value, tree=tree.value,
        measurement=measurement.value,
        measurement_type=method.value,
        optics=False
        # optics = use_optics.value
        )
    if data_object.is_optics_available and use_optics.value == True:
        data_object = static_pull.DynatreeStaticMeasurement(
            day=day.value, tree=tree.value,
            measurement=measurement.value,
            measurement_type=method.value,
            optics=True)
    return data_object

@task
def nakresli(reset_measurements=False):
    start = time.time_ns()/1000000
    # lib_dynatree.logger.info(f"Is {day.value} {tree.value} {measurement.value} nakresli current? {nakresli.is_current()}")
    if not nakresli.is_current():
        # lib_dynatree.logger.info("Interrupting non current function nakresli")
        nakresli.cancel()
        return None
    data_object = get_data_object()
    figs = [data_object.plot()] 
    if include_details.value:
        figs = figs + [i.plot(n) for n,i in enumerate(data_object.pullings)]
    end = time.time_ns()/1000000
    # print(f"nakresli took {end-start}ms.")
    return figs

styles_css = """
        .widget-image{width:100%;} 
        .v-btn-toggle{display:inline;}  
        .v-btn {display:inline; text-transform: none;} 
        .vuetify-styles .v-btn-toggle {display:inline;} 
        .v-btn__content { text-transform: none;}
        """

# first_pass = True
# http://localhost:8765/tahovky?tree=BK04&method=normal&measurement=M02&use_optics=True&day=2022-08-16
# http://localhost:8765/tahovky?tree=BK08&method=den&measurement=M03&use_optics=False&day=2022-08-16

@solara.component
def Page():
    # global first_pass
    
    # # if first_pass:
    # #     first_pass = False
    # if data_from_url.value:
    #     args =  solara.use_router().search
    #     if len(str(args))>0:
    #         parsed = str(args).split("&")
    #         params = {}
    #         for i in parsed:
    #             _,__ = i.split("=")
    #             params[_] = __
    #         solara.Text(str(params))
    #         tree.value = params['tree']
    #         day.value = params['day']
    #         method.value = params['method']
    #         measurement.value  =params['measurement']
    #         use_optics.value = params['use_optics']=="True"
        
    solara.Title(title)
    solara.Style(styles_css)
    with solara.Sidebar():
        # solara.Markdown("Výběr proměnných pro záložku \"Volba proměnných a regrese\".")
        Selection()
    # solara.Markdown("# Under construction")
    # return
    with solara.lab.Tabs():
        with solara.lab.Tab("Grafy"):
            with solara.Card():
                try:
                    start = time.time_ns()/1000000
                    Graphs()
                    end = time.time_ns()/1000000
                    # print(f"Graphs took {end-start}ms.")
                except:
                    pass
        with solara.lab.Tab("Volba proměnných a regrese"):
            with solara.Card(title="Increasing part of the time-force diagram"):
                try:
                    start = time.time_ns()/1000000
                    Detail()
                    end = time.time_ns()/1000000
                    # print(f"Details took {end-start}ms.")
                except:
                    pass
        with solara.lab.Tab("Statistiky"):
            with solara.Card():
                try:
                    start = time.time_ns()/1000000
                    Statistics()
                    end = time.time_ns()/1000000
                    # with solara.AppBar():
                    #     solara.Text(f"Statistics took {end-start}ms.")
                except:
                    pass

        with solara.lab.Tab("Návod a komentáře"):
            with solara.Card(title="Návod"):
                Help()

    # MainPage()

# def clear():
#     # https://stackoverflow.com/questions/37653784/how-do-i-use-cache-clear-on-python-functools-lru-cache
#     static_pull.nakresli.__dict__["__wrapped__"].cache_clear()
#     static_pull.process_data.__dict__["__wrapped__"].cache_clear()

@solara.component
def Selection():
    with solara.Card(title="Measurement choice"):
        with solara.Column():
            # solara.Switch(label="Use data from URL", value=data_from_url)
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
        with solara.Tooltip("Umožní použít preprocessing udělaný na tahovkách M02 a více. Tím sice nebude stejná metodika jako pro M01 (tam se preprocessing nedělal), ale máme časovou synchronizaci s optikou, o opravu vynulování inklinometrů a pohyb bodů Pt3 a Pt4."):
            solara.Switch(
                label="Use optics data, if possible",
                value=use_optics,
                disabled=not data_object.is_optics_available,
                on_value=nakresli
            )
        # solara.Div(style={"margin-bottom": "10px"})
        solara.Button("Run calculation", on_click=nakresli, color="primary")
        # solara.Button("Clear cache", on_click=clear(), color="primary")
        solara.Markdown(
            f"**Selected**: {day.value}, {tree.value}, {measurement.value}")
        if data_object.is_optics_available:
            solara.Markdown("✅ Optics is available for this measurement.")
        else:
            solara.Markdown(
                "❎ Optics is **not** available for this measurement.")

def Statistics():
    data_object = get_data_object()
    if data_object.is_optics_available:
        l = [data_object.data_optics_extra, data_object.data_pulling]
        titles = ["Pulling data interpolated to optics time", "Pulling data"]
    else:
        l = [data_object.data_pulling]
        titles = ["Pulling data"]
    with solara.Columns([6,6]):
        for df,title in zip(l,titles):
            with solara.Card():
                solara.Markdown(f"**{title}**")
                df = df[[i for i in df.columns if "fixed" not in i[0]]]
                nans = pd.DataFrame(df.isna().sum())
                nans.loc[:,"name"] = df.columns
                nans.columns = ["#nan", "name"]
                nans = nans[["name","#nan"]]
                solara.Markdown(f"Shape: {df.shape}")
                solara.DataFrame(nans)
    # try:
    #     solara.DataFrame(pd.concat([pd.DataFrame(subdf.index),subdf], axis=1))
    # except:
    #     pass

@solara.component
def Graphs():
    solara.ProgressLinear(nakresli.pending)
    if measurement.value not in available_measurements(df.value, day.value, tree.value, method.value):
        with solara.Error():
            solara.Markdown(
                f"""
                * Measurement {measurement.value} not available for tree {tree.value}
                  day {day.value} measurement type {method.value}.
                * You may need to switch measurement type (normal/den/noc/...) 
                  if the list of the measuemrent day is incorrect.
                """)
        return

    if nakresli.not_called:
        solara.Info(solara.Markdown("""
                        
* Vyber měření a stiskni tlačítko \"Run calculation\". 
* Ovládací prvky jsou v sidebaru. Pokud není otevřený, otevři kliknutím na tři čárky nalevo v modrém pásu.
* Při změně vstupů se většinou obrázek aktualizuje, ale ne vždy. Pokud nadpis na obrázku nesouhlasí s vybranými hodnotami, spusť výpočet tlačítkem \"Run calculation\".

                                    """))
        # solara.Warning(
        #     "Pokud pracuješ v prostředí JupyterHub, asi bude lepší aplikaci maximalizovat. Tlačítko je v modrém pásu úplně napravo.")
    elif not nakresli.finished:
        with solara.Row():
            solara.Markdown("""
            * Pracuji jako ďábel. Může to ale nějakou dobu trvat. 
            * Pokud to trvá déle jak 10 sekund, vzdej to, zapiš si měření a zkusíme zjistit proč. Zatím si prohlížej jiná měření.
            """)
            solara.SpinnerSolara(size="100px")
    else:
        solara.Markdown("Na obrázku je průběh experimentu (časový průběh síly). Volitelně můžeš zobrazit detaily pro rozmezí 30%-90% maxima síly. \n\n V detailech je časový průběh síly, časový průběh na inklinometrech a grafy inklinometry versus síla nebo moment.")
        with solara.Tooltip("Allows to show details of the pulls on this page. Slows down the computation, however."):
            solara.Switch(label="Show details", value=include_details, on_value=nakresli)
        f = nakresli.value
        if include_details.value:
            ncols = 6
        else:
            ncols = 12
        with solara.ColumnsResponsive(ncols):
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

msg = """
### Něco se nepovedlo. 
                     
* Možná není vybráno nic pro svislou osu. 
* Možná je vybrána stejná veličina pro vodorovnou a svislou osu. 
* Nebo je nějaký jiný problém. Možná mrkni nejprve na záložku Grafy."""

def Detail():
    global subdf
    if nakresli.not_called:
        solara.Info(
            "Nejdřív nakresli graf v první záložce. Klikni na Run calculation v sidebaru.")
        return
    if not nakresli.finished:
        with solara.Row():
            solara.Text("Pracuji jako ďábel. Může to ale nějakou dobu trvat.")
            solara.SpinnerSolara(size="100px")
            return
    solara.Markdown("""
            Pro výběr proměnných na vodorovnou a svislou osu otevři menu v sidebaru (tři čárky v horním panelu). Po výběru můžeš sidebar zavřít. Přednastavený je moment vypočítaný z pevného naměřeného úhlu lana na vodorovné ose a oba inlinometry na svislé ose.
            """)

    with solara.Sidebar():
        cols = ['Time', 'Pt3', 'Pt4', 'Force(100)', 'Elasto(90)', 'Elasto-strain',
                # 'Inclino(80)X', 'Inclino(80)Y', 'Inclino(81)X', 'Inclino(81)Y',
                'RopeAngle(100)',
                'blue', 'yellow', 'blueX', 'blueY', 'yellowX', 'yellowY',
                'blue_Maj', 'blue_Min', 'yellow_Maj', 'yellow_Min',
                'F_horizontal_Rope', 'F_vertical_Rope',
                'M_Rope', 'M_Pt_Rope', 'M_Elasto_Rope',
                'F_horizontal_Measure', 'F_vertical_Measure',
                'M_Measure', 'M_Pt_Measure', 'M_Elasto_Measure',]
        with solara.Card(title="Horizontal axis"):
            solara.Markdown("Choose one variable.")
            solara.ToggleButtonsSingle(values=cols, value=xdata, dense=True)
        with solara.Card(title="Vertical axis"):
            solara.Markdown(
                "Choose one or more variables. You cannot choose the same variable which has been used for horizontal axis.")
            solara.ToggleButtonsMultiple(
                values=cols[1:], value=ydata, dense=True)
        with solara.Card(title="Second vertical axis"):
            with solara.VBox():
                with solara.Tooltip("Choose one variable for second vertical axis, shown on the right. (Only limited support in interactive plots. In interactive plots we plot rescaled data. The scale factor is determined from maxima.) You cannot choose the variable used for horizontal axis."):
                    with solara.VBox():
                        solara.Text("🛈 (hover here for description)")

                solara.ToggleButtonsSingle(
                    values=[None]+cols[1:], value=ydata2, dense=True)
    with solara.Row():

        if not use_optics.value:
            if ydata2.value in ["Pt3", "Pt4"]:
                ydata2.value = None
                return
            if xdata.value in ["Pt3", "Pt4"]:
                xdata.value = "Time"
                return
            new = [i for i in ydata.value if i not in ["Pt3", "Pt4"]]
            if new != ydata.value:
                ydata.value = new
                return
        
        temp_data_object = static_pull.DynatreeStaticMeasurement(
            day=day.value, tree=tree.value,
            measurement=measurement.value, measurement_type=method.value,
            optics=False)
        if measurement.value == "M01":
            with solara.Card():
                solara.Markdown("**Pull No. of M01:**")
                with solara.Column(**tightcols):
                    pulls = list(range(len(temp_data_object.pullings)))
                    solara.ToggleButtonsSingle(values=pulls, value=pull)
                pull_value = pull.value
        else:
            pull_value = 0
        if (use_optics.value) and (not temp_data_object.is_optics_available):
            use_optics.value = False
            return

        with solara.Card():
            solara.Markdown("**Bounds to cut out boundaries in % of Fmax**")
            with solara.Column(**tightcols):
                solara.ToggleButtonsSingle(
                    values=data_possible_restrictions, value=restrict_data)
        with solara.Card():
            with solara.Column(**tightcols):
                with solara.Tooltip("Umožní zobrazit graf pomocí knihovny Plotly. Bude možné zoomovat, odečítat hodnoty, klikáním na legendu skrývat a odkrývat proměnné apod. Nebudou zobrazeny regresní prímky."):
                    solara.Switch(label="Interactive graph",
                                  value=interactive_graph)
                with solara.Tooltip("Umožní zobrazit grafy veličin pro celý časový průběh."):
                    solara.Switch(
                        label="Ignore time restriction", value=all_data)
                    
    if restrict_data.value == data_possible_restrictions[0]:
        restricted = None
    elif restrict_data.value == data_possible_restrictions[1]:
        restricted = (0.1, 0.9)
    else:
        restricted = (0.3, 0.9)

    d_obj = static_pull.DynatreeStaticMeasurement(
        day=day.value, tree=tree.value,
        measurement=measurement.value, 
        measurement_type=method.value,
        optics=use_optics.value, 
        restricted=restricted)
    dataset = d_obj.pullings[pull_value]
    subdf = dataset.data
    if all_data.value:
        _ = d_obj._get_static_pulling_data(optics=use_optics.value, restricted='get_all')
        _["Time"] = _.index
        subdf = static_pull.DynatreeStaticPulling(_, tree=tree.value)
        subdf = subdf.data
    
    try:
        # find regresions
        if (xdata.value != "Time") and not all_data.value:
            # subsubdf = subdf.loc[:,[xdata.value]+[i for i in ydata.value+[ydata2.value] if i!=xdata.value]]
            ydata.value = [i for i in ydata.value if i != xdata.value]
            if xdata.value == ydata2.value:
                ydata2.value = None
            if ydata2.value is None:
                target = ydata.value
            else:
                target = ydata.value + [ydata2.value]
            reg_df = static_pull.DynatreeStaticPulling._get_regressions(subdf, [[xdata.value]+target])
            solara.DataFrame(reg_df.iloc[:, :5])
            # solara.display(reg_df.iloc[:, :5])
            df_subj_reg = subdf[[xdata.value]+target]
        else:
            solara.Info(
                """
                No regressions if independent variable is Time or if all
                data are considered. (Put "Ignore time restriction off and select another independent variable.")
                """)
    except:
        solara.Error(
            "Něco se pokazilo při hledání regresí. Nahlaš prosím problém. Pro další práci vyber jiné veličiny.")

    title = f"{day.value} {tree.value} {measurement.value} {method.value} Pull {pull_value}"
    if interactive_graph.value:
        kwds = {"template": "plotly_white", "height": 600, "title": title}
        try:
            if ydata2.value != None:  # Try to add rescaled column
                maximum_target = np.nanmax(
                    subdf[fix_input(ydata.value)].values)
                maximum_ori = np.nanmax(subdf[ydata2.value].values)
                subdf.loc[:, f"{ydata2.value}_rescaled"] = subdf.loc[:,
                                                                      ydata2.value]/np.abs(maximum_ori/maximum_target)
                extradata = [f"{ydata2.value}_rescaled"]
            else:
                extradata = []
            cols_to_draw = fix_input(ydata.value + extradata)
            if xdata.value == "Time":
                px.scatter(subdf, y=cols_to_draw, **kwds)
            else:
                px.scatter(subdf, x=xdata.value, y=cols_to_draw, **kwds)
        except:
            solara.Error(solara.Markdown(
                """### Image failed. 
                         
                * Something is wrong. Switch to noninteractive plot or change variables setting. 
                * This error appears especially if you try to plot both forces and inclinometers on vertical axis.
                """))
        pass
    else:
        fig, ax = plt.subplots()
        if xdata.value == "Time":
            subdf["Time"] = subdf.index
        subdf.plot(x=xdata.value, y=ydata.value, style='.', ax=ax)
        try:
            if xdata.value != "Time":
                t = np.linspace(*ax.get_xlim(), 5)
                for y in ydata.value:
                    if y not in reg_df["Dependent"]:
                        continue
                    d = reg_df[reg_df["Dependent"] ==
                               y].loc[:, ["Slope", "Intercept"]]
                    ax.plot(t, t*d.iat[0, 0]+d.iat[0, 1],
                            **regression_settings)
        except:
            pass
        if ydata2.value != None:
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
        ax.set(title=title)
        with solara.Card(
                style={"max-width": "1000px"}
                ):
            solara.FigureMatplotlib(fig)
    try:
        with solara.Card():
            solara.Text("Data used for the regressions")
            solara.DataFrame(df_subj_reg)
    except:
        pass


def Help():
    solara.Markdown(
        """
### Práce

* Vyber datum, strom a měření. Pokud se obrázek neaktualizuje automaticky, klikni na tlačítko pro spuštění výpočtu. Výpočet se spustí kliknutím tlačítka nebo změnou volby měření. Pokud se mění strom nebo den a měření zůstává M01, je potřeba stisknout tlačítko.
* Zobrazí se průběh experimentu, náběh (resp. tři náběhy) síly do maxima a zvýrazněná část pro analýzu. Ta je 30-90 procent, ale dá se nastavit i 10-90 procent nebo 0-100 procent.
* Je možné ignorovat omezení a vykreslit celý průběh experimentu. To má smysl asi jenom u M01
* Poté máš možnost si detailněji vybrat, co má být v dalším grafu na vodorovné a svislé ose. Tlačítka pro výběr se objeví v bočním panelu, aby se dala skrývat a nezavazela. Počítá se regrese mezi veličinou na vodorovné ose a každou z veličin na ose svislé. Regrese nejsou dostupné, pokud je vodorovně čas (nedávalo by smysl) a pokud je na vodorovné a svislé ose stejná veličina (taky by nedávalo smysl).

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
* Graf moment versus inklinometry má někdy na začátku trochu neplechu. Možná mají velký vliv nevynulované hodnoty 
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

Pokud chceš dynamické měření brát jako statiku, použij přepínač "Use optics data, if possible" (pod seznamem dostupných měření.)

### Historie

* 2024-08-??: první verze
* 2024-08-23: je možné volit ořez dat v procentech Fmax mezi 0-100%, 10%-90% a 30%-90%, zobrazuje se regresní přímka. TODO: najít v datech, kde se to nejvíce liši a nechat info zde.
* 2024-08-25: zobrazují se i data, ke kterým není nebo zatím není optika, vylepšení ovládání, většinou se výpočet spouští automaticky při změně parametrů
* 2024-08-26: bereme do úvahy i den/noc/afterro/mraz
* 2024-08-28: přepsáno pomocí tříd a OOP, mírná blbuvzdornost při volbě proměnných na osy s detailním grafem, kontrola dosupnosti optiky se promítá i do přepínačů
* 2024-08-29: zařazeno pod jednu střechu s dalšími aplikacemi
"""
    )

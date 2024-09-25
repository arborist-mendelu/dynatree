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
import plotly.express as plx
import solara.lab
from solara.lab import task
import solara
import time
import solara_select_source as s
import graphs_regressions
import static_lib_pull_comparison
import plotly.graph_objects as go
DATA_PATH = "../data"

tightcols = {'gap': "0px"}
regression_settings = {'color': 'gray', 'alpha': 0.5}

title = "DYNATREE: pulling, force, inclinometers, extensometer, optics, ..."

include_details = solara.reactive(False)

# Create data object when initialized
data_object = lib_dynatree.DynatreeMeasurement(
    s.day.value, 
    s.tree.value, 
    s.measurement.value,
    measurement_type=s.method.value,
    # use_optics=use_optics.value
    )

data_possible_restrictions = ["0-100%", "10%-90%", "30%-90%"]

xdata = solara.reactive("M")
ydata = solara.reactive(["blue", "yellow"])
ydata2 = solara.reactive([])
pull = solara.reactive(0)
restrict_data = solara.reactive(data_possible_restrictions[-1])
interactive_graph = solara.reactive(False)
all_data = solara.reactive(False)
force_interval = solara.reactive("None")
tab_index = solara.reactive(0)

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
    return nakresli()

# def get_data_object(day, tree, measuemrent, measurement_type, optics):
def get_data_object():
    """
    Return the measurements. 
    If optics is not available, force optics to False.
    
    Cheap, 0.0002s.
    """
    data_object = static_pull.DynatreeStaticMeasurement(
        day=s.day.value, tree=s.tree.value,
        measurement=s.measurement.value,
        measurement_type=s.method.value,
        optics=False
        # optics = use_optics.value
        )
    if data_object.is_optics_available and s.use_optics.value == True:
        data_object = static_pull.DynatreeStaticMeasurement(
            day=s.day.value, tree=s.tree.value,
            measurement=s.measurement.value,
            measurement_type=s.method.value,
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
def prehled():
    solara.Markdown(
"""
* Tady jsou směrnice z regresí M/blue, M/yellow a M_Elasto/Elasto. Pokud nějaká
  hodnota ulítává, je možné, že inklinometr nebo extenzometr špatně měřil. V takovém případě se 
  kontroluje asi časový průběh příslušného přístroje.
* True/False se vztahuje k přítomnosti listů. 
* Číslo 0 až 2 se vztahuje k počtu ořezů.
* V sidebaru vlevo můžeš přepínat strom, graf by se měl automaticky aktualizovat.
"""
        )
    # with solara.Row():
    #     solara.Button("Update Page", on_click=ShowRegressionsHere)
    images = graphs_regressions.main(trees=[s.tree.value])
    df_failed = pd.read_csv("csv/static_fail.csv")
    df_checked = pd.read_csv("csv/static_checked_OK.csv")
    for t,f in images.items():
        with solara.Card():
            solara.FigurePlotly(f)
            solara.Markdown(f"Failed experiments")
            solara.display(df_failed[df_failed["tree"]==t])
            solara.Markdown(f"Succesfully checked experiments")
            solara.display(df_checked[df_checked["tree"]==t])
    solara.FileDownload(graphs_regressions.read_data().to_csv(), filename="static_dynatree.csv", label="Download data")

@solara.component
def normalized_slope():
    df_merged = static_lib_pull_comparison.df_merged
    subdf = df_merged[df_merged["pullNo"]!=0].loc[:,
        ["pullNo","Slope_normalized","tree", "Dependent","type","day"]]
    subdf = subdf[subdf["tree"]==s.tree.value].sort_values(by="day")
    cat_order = subdf["day"].drop_duplicates().tolist()
    fig = plx.box(
        subdf, 
        x="day", 
        y="Slope_normalized", 
        color='type', 
        points='all', 
        hover_data=["tree","type","pullNo", "Dependent"],
        category_orders={"day": cat_order},
        height = s.height.value, width=s.width.value,
        title=f"Tree {s.tree.value}", 
        template =  "plotly_white", 
        )
    fig.update_layout(xaxis=dict(type='category'))
    solara.FigurePlotly(fig)    

@solara.component
def Page():
        
    solara.Title(title)
    solara.Style(styles_css)
    with solara.Sidebar():
        Selection()
        if tab_index.value in [3,4]:
            s.ImageSizes()
            s.width.value = 1200
    with solara.lab.Tabs(value=tab_index):
        with solara.lab.Tab("Grafy"):
            with solara.Card():
                try:
                    if tab_index.value == 0:
                        Graphs()
                except:
                    pass
        with solara.lab.Tab("Volba proměnných a regrese"):
            with solara.Card(title="Increasing part of the time-force diagram"):
                try:
                    if tab_index.value == 1:
                        Detail()
                except:
                    pass
        with solara.lab.Tab("Polární graf"):
            with solara.Card():
                try:
                    if tab_index.value == 2:
                        Polarni()
                except:
                    pass

        with solara.lab.Tab("Srovnání s prvním zatáhnutím"):
            with solara.Card():
                solara.Markdown(
"""
* V grafech je podíl směrnice z druhého nebo třetího zatáhnutí  směrnice z prvního zatáhnutí. Toto je v grafu vedeno jako Slope_normalized.
* Pokud věříme, že při první zatáhnutí je systém tužší, měl by podíl být stabilně pod jedničkou.
* V sidebaru vlevo můžeš přepínat strom, graf by se měl automaticky aktualizovat.
""")
                try:
                    if tab_index.value == 3:
                        normalized_slope()
                except:
                    pass

                    # solara.FigurePlotly(figPl)                
        with solara.lab.Tab("Přehled"):
            with solara.Column():
                try:
                    if tab_index.value == 4:
                        prehled()
                except:
                    pass
        with solara.lab.Tab("Komentáře & dwnl."):
            with solara.Card(title="Návod"):
                Help()

@solara.component
def Selection():
    s.Selection()
    data_object = lib_dynatree.DynatreeMeasurement(
        s.day.value, s.tree.value, s.measurement.value,measurement_type=s.method.value)
    with solara.Column(align='center'):
        solara.Button("Run calculation", on_click=nakresli, color="primary")

def fixdf(df):
    df.columns = [f"{i[0]}" if i[1]=='nan' else f"{i[0]}_{i[1]}" for i in df.columns]
    df = df[[i for i in df.columns if "_" not in i]]
    return df

def Statistics():
    data_object = get_data_object()
    solara.Markdown(
"""
This card reports missing data.

* Rope(100) is never used
* Inclino(80) and Inclino(81) are claculated from the other data
""")
    if data_object.is_optics_available:
        l = [fixdf(data_object.data_optics_extra), data_object.data_pulling]
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
    if s.measurement.value not in available_measurements(s.df.value, s.day.value, s.tree.value, s.method.value):
        with solara.Error():
            solara.Markdown(
                f"""
                * Measurement {s.measurement.value} not available for tree {s.tree.value}
                  day {s.day.value} measurement type {s.method.value}.
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

def Polarni():
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

    with solara.Row():
        temp_data_object = static_pull.DynatreeStaticMeasurement(
            day=s.day.value, tree=s.tree.value,
            measurement=s.measurement.value, measurement_type=s.method.value,
            optics=False)
        if s.measurement.value == "M01":
            with solara.Card():
                solara.Markdown("**Pull No. of M01:**")
                with solara.Column(**tightcols):
                    pulls = list(range(len(temp_data_object.pullings)))
                    solara.ToggleButtonsSingle(values=pulls, value=pull)
                pull_value = pull.value
        else:
            pull_value = 0
        with solara.Card():
            solara.Markdown("**Bounds to cut out boundaries in % of Fmax**")
            with solara.Column(**tightcols):
                solara.ToggleButtonsSingle(
                    values=data_possible_restrictions, value=restrict_data)
        
        if restrict_data.value == data_possible_restrictions[0]:
            restricted = None
        elif restrict_data.value == data_possible_restrictions[1]:
            restricted = (0.1, 0.9)
        else:
            restricted = (0.3, 0.9)

    d_obj = static_pull.DynatreeStaticMeasurement(
        day=s.day.value, tree=s.tree.value,
        measurement=s.measurement.value, 
        measurement_type=s.method.value,
        optics=s.use_optics.value, 
        restricted=restricted)
    dataset = d_obj.pullings[pull_value]
    subdf = dataset.data
    fig,ax = plt.subplots()
    ax.plot(subdf['blueMaj'],subdf['blueMin'], label="blue")
    ax.plot(subdf['yellowMaj'],subdf['yellowMin'], label="yellow")
    ax.set_aspect('equal')
    ax.legend()
    bound = [*ax.get_xlim(), *ax.get_ylim()]
    bound = np.abs(np.array(bound)).max()
    title = f"{d_obj.measurement_type} {d_obj.day} {d_obj.tree} {d_obj.measurement}"
    if d_obj.measurement == "M01":
        title = f"{title}, PullNo {pull_value}"
    ax.set(xlim=(-bound,bound), ylim=(-bound,bound), title=title, xlabel="Major inclinometer", ylabel="Minor inclinometer")
    ax.grid(which='both')

    solara.FigureMatplotlib(fig)

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
                'blue', 'yellow',
                'blueMaj', 'blueMin', 'yellowMaj', 'yellowMin',
                'F_horizontal', 'F_vertical',
                'M', 'M_Pt', 'M_Elasto',
                ]
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

        if not s.use_optics.value:
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
            day=s.day.value, tree=s.tree.value,
            measurement=s.measurement.value, measurement_type=s.method.value,
            optics=False)
        if s.measurement.value == "M01":
            with solara.Card():
                solara.Markdown("**Pull No. of M01:**")
                with solara.Column(**tightcols):
                    pulls = list(range(len(temp_data_object.pullings)))
                    solara.ToggleButtonsSingle(values=pulls, value=pull)
                pull_value = pull.value
        else:
            pull_value = 0
        if (s.use_optics.value) and (not temp_data_object.is_optics_available):
            s.use_optics.value = False
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
        day=s.day.value, tree=s.tree.value,
        measurement=s.measurement.value, 
        measurement_type=s.method.value,
        optics=s.use_optics.value, 
        restricted=restricted)
    dataset = d_obj.pullings[pull_value]
    subdf = dataset.data
    if all_data.value:
        _ = d_obj._get_static_pulling_data(optics=s.use_optics.value, restricted='get_all')
        _["Time"] = _.index
        subdf = static_pull.DynatreeStaticPulling(_, tree=s.tree.value, measurement_type=s.method.value,  extra_columns={"blue":"Inclino(80)", "yellow":"Inclino(81)",
        **d_obj.identify_major_minor})
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
            reg_df = static_pull.DynatreeStaticPulling._get_regressions(subdf, [[xdata.value]+target], )
            solara.DataFrame(reg_df.iloc[:, :5])
            # solara.display(reg_df.iloc[:, :5])
            df_subj_reg = subdf[[xdata.value]+target]
        else:
            solara.Info(
                """
                No regressions are calculated if the independent variable is Time or if all
                data are considered. (Put "Ignore time restriction off and select another independent variable.")
                """)
    except:
        solara.Error(
            "Něco se pokazilo při hledání regresí. Nahlaš prosím problém. Pro další práci vyber jiné veličiny. Pokud tato hláška během chvíle zmizí, je neškodná.")

    title = f"{s.day.value} {s.tree.value} {s.measurement.value} {s.method.value} Pull {pull_value}"
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
    plt.close('all')

def stahni_csv(file, label="Download", msg=None):
    with solara.Row():
        solara.FileDownload(
            pd.read_csv(file).to_csv(index=None), 
            label=label,
            filename=file.split("/")[-1])
        solara.Text(msg)
    
def Help():
    with solara.Card():
        with solara.Column():
            solara.Text("Použité parametry")
            stahni_csv("csv/static_fail.csv", msg="Zkoušky klasifikované jako nepovedené")
            stahni_csv("csv/static_checked_OK.csv", msg="Zkoušky klasifikované jako OK, i když se hodnoty liší od ostatních")
            stahni_csv("csv/reset_inclinometers.csv", msg="Ručně vynulované inklinometry")
    solara.Markdown(
        """
### Práce

* Vyber datum, strom a měření. Pokud se obrázek neaktualizuje automaticky, klikni na tlačítko pro spuštění výpočtu. Výpočet se spustí kliknutím tlačítka nebo změnou volby měření. Pokud se mění strom nebo den a měření zůstává M01, je potřeba stisknout tlačítko.
* Zobrazí se průběh experimentu, náběh (resp. tři náběhy) síly do maxima a zvýrazněná část pro analýzu. Ta je 30-90 procent, ale dá se nastavit i 10-90 procent nebo 0-100 procent.
* Je možné ignorovat omezení a vykreslit celý průběh experimentu. To má smysl asi jenom u M01
* Poté máš možnost si detailněji vybrat, co má být v dalším grafu na vodorovné a svislé ose. Tlačítka pro výběr se objeví v bočním panelu, aby se dala skrývat a nezavazela. Počítá se regrese mezi veličinou na vodorovné ose a každou z veličin na ose svislé. Regrese nejsou dostupné, pokud je vodorovně čas (nedávalo by smysl) a pokud je na vodorovné a svislé ose stejná veličina (taky by nedávalo smysl).

### Popis

* Inlinometr blue je 80, yelllow je 81. Výchylky v jednotlivých osách jsou blueX a blueY resp. blueMaj a blueMin. Celková výchylka je blue. Podobně  druhý inklinometr.
* F se rozkládá na vodorovnou a svislou složku.Vodorovná se používá k výpočtu momentu v bodě úvazu (M), v bodě Pt3 (M_Pt) a v místě s extenzometrem (M_Elasto). 
* Elasto-strain je Elasto(90)/200000.

### Komenáře

* V diagramech síla nebo moment versus inklinometry není moc změna trendu mezi první polovinou diagramu a celkem. Takže je asi jedno jestli bereme pro sílu rozmezí 10-90 procent Fmax nebo 10-40 procent.
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
* 2024-09-?? polární graf, interaktivní grafy
"""
    )

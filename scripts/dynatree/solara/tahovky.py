#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:00:04 2024

@author: DYNATREE project, ERC CZ no. LL1909 "Tree Dynamics: Understanding of Mechanical Response to Loading"
"""

import time
from dynatree.find_measurements import available_measurements
from dynatree import static_pull, dynatree_util as du
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rich
# import glob
import solara.express as px
import plotly.express as plx
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import solara.lab
from solara.lab import task
import solara
from solara.lab.components.confirmation_dialog import ConfirmationDialog
import dynatree.solara.select_source as s
import graphs_regressions
import static_lib_pull_comparison
import dynatree.dynatree as dynatree
from solara.lab import task

import logging
dynatree.logger.setLevel(dynatree.logger_level)

loading_start = time.time()

DATA_PATH = "../data"
from great_tables import GT, style, loc
import seaborn as sns
# from weasyprint import HTML, CSS
# import logging
# lib_dynatree.logger.setLevel(logging.INFO)

import config

cmap = {
    'normal': 'steelblue',
    'den': 'cadetblue',
    'noc': 'black', 
    'afterro':'purple', 
    'afterro2':'orange',
    'mokro':'blue',
    'mraz':'red',
    }

tightcols = {'gap': "0px"}
regression_settings = {'color': 'gray', 'alpha': 0.5}

title = "DYNATREE: pulling, force, inclinometers, extensometer, optics, ..."

include_details = solara.reactive(False)

# Create data object when initialized
data_object = dynatree.DynatreeMeasurement(
    s.day.value,
    s.tree.value,
    s.measurement.value,
    measurement_type=s.method.value,
    # use_optics=use_optics.value
)

data_possible_restrictions = ["0-100%", "10%-90%", "30%-90%"]

xdata = solara.reactive("M")
ydata = solara.reactive(["blueMaj", "yellowMaj"])
ydata2 = solara.reactive([])
pull = solara.reactive(0)
restrict_data = solara.reactive(data_possible_restrictions[-1])
interactive_graph = solara.reactive(False)
all_data = solara.reactive(False)
force_interval = solara.reactive("None")
tab_index = solara.reactive(0)
subtab_index = solara.reactive(0)
subtab_indexB = solara.reactive(0)
include_statics = solara.reactive(True)
include_dynamics = solara.reactive(True)


# data_from_url = solara.reactive(False)
# 
def fix_input(a):
    """
    The input is a list. 

    The output is ["Force(100)"] if the input is empty. If not empty, the 
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
    # start = time.time_ns()/1000000
    # lib_dynatree.logger.info(f"Is {day.value} {tree.value} {measurement.value} nakresli current? {nakresli.is_current()}")
    if not nakresli.is_current():
        # lib_dynatree.logger.info("Interrupting non current function nakresli")
        nakresli.cancel()
        return None
    data_object = get_data_object()
    if data_object.file_pulling_name is None:
        return None
    figs = [data_object.plot()]
    if include_details.value:
        figs = figs + [i.plot(n) for n, i in enumerate(data_object.pullings)]
    # end = time.time_ns()/1000000
    # print(f"nakresli took {end-start}ms.")
    return figs

# first_pass = True
# http://localhost:8765/tahovky?tree=BK04&method=normal&measurement=M02&use_optics=True&day=2022-08-16
# http://localhost:8765/tahovky?tree=BK08&method=den&measurement=M03&use_optics=False&day=2022-08-16
def click_info():
    solara.Info(solara.Markdown( """
* Kliknutí na tečku v grafu zobrazí časový průběh na senzoru a  časový průběh síly. Data jsou dointerpolovaná, 
  rovná čára znamená pravděpodobně chybějící dointerpolovaná data.
* Se **shiftem** kreslí pomocí bodů, jinak pomocí čar. S **ctrl** kreslí celý experiment a data nejsou dointerpolovaná.
  S **alt** se kreslí body použité pro regresi, vodorovně je síla, svisle data ze senzoru.
* Výška obrázku je podle volby v levém sloupci, šířka je přes celé okno.
* V obrázcích s trendem nejsou ručně vyhozená data (červeně v obrázku "hledání odlehlých").
""", style={'color': 'inherit'}))

def info_figure():
    return
    solara.Info("Ctrl click pro celý experiment, Shift+click pro tečky, Ctrl+Shift+click pro obojí.")

@solara.component
def prehled():
    global figdata

    with solara.Sidebar():
        limitR2()
        limit_statics_dynamics()
    click_info()
        # with solara.Row():
    #     solara.Button("Update Page", on_click=ShowRegressionsHere)
    images = graphs_regressions.main(trees=[s.tree.value], width=s.width.value, height=s.height.value,
                                     limitR2=[R2limit_lower.value, R2limit_upper.value],
                                     include_statics=include_statics.value, include_dynamics=include_dynamics.value)
    df_failed = pd.read_csv(config.file['static_fail'])
    df_checked = pd.read_csv(config.file['static_checked_OK'])
    overlay(on_click_more)

    for t, f in images.items():
        with solara.Card():
            figdata = f
            solara.FigurePlotly(f, on_click=on_click_more)
            solara.Markdown(f"Failed experiments")
            solara.display(df_failed[df_failed["tree"] == t])
            solara.Markdown(f"Succesfully checked experiments")
            solara.display(df_checked[df_checked["tree"] == t])
    solara.FileDownload(graphs_regressions.read_data().to_csv(), filename="static_dynatree.csv", label="Download data")
    with solara.Info():
        solara.Markdown(
            """
            **Přehled dat pro jednotlivé veličiny a stromy**

            * Tady jsou směrnice z regresí M/Inclinometr a M_Elasto/Elasto. Pokud nějaká
              hodnota ulítává, je možné, že inklinometr nebo extenzometr špatně měřil. V takovém případě se 
              kontroluje asi časový průběh příslušného přístroje.
            * True/False se vztahuje k přítomnosti listů. 
            * Číslo 0 až 2 se vztahuje k počtu ořezů.
            * V sidebaru vlevo můžeš přepínat strom, graf by se měl automaticky aktualizovat.
            * Ručně vyřazené experimenty jsou v obrázku červeně a jsou v tabulce pod obrázkem. Odlehlé experimenty, 
              které byly ručně zkontorlovány a uznány jako OK jsou v tabulce pod vyřazenými.
            """, style={'color': 'inherit'}
        )

sort_ascending = solara.reactive(True)
show_dialog = solara.reactive(False)

def close_dialog():
    show_dialog.value = False

info = solara.reactive("")
@task
def on_click(event, kwds=None):
    if kwds is None:
        kwds = {i: event['device_state'][i] for i in ['ctrl', 'shift', 'alt']}
    group = event['points']['trace_indexes'][0]
    position = event['points']['point_indexes'][0]
    data = figdata['data'][group]['customdata'][position]
    return {'fig': click_figure(data, event, **kwds),
            'event':event, 'kwds': kwds}

@task
def on_click_more(event, kwds=None):
    if kwds is None:
        kwds = {i: event['device_state'][i] for i in ['ctrl', 'shift', 'alt']}
    group = event['points']['trace_indexes'][0]
    position = event['points']['point_indexes'][0]
    data = figdata['data'][group]['customdata'][position]
    day, tree, measurement, mt, pullNo, R, remark, indep, dep,  camera = list(data)
    data = [tree, mt, pullNo, indep, dep, measurement, camera, R, day ]
    return {'fig': click_figure(data, event, **kwds),
            'event':event, 'kwds': kwds}

def overlay(task):
    with ConfirmationDialog(show_dialog.value, on_ok=close_dialog, on_cancel=close_dialog, max_width  = '100%',
                            title=""):
        solara.ProgressLinear(task.pending)
        if task.finished:
            ans = task.value['fig']
            solara.FigurePlotly(ans)
            info_figure
            # solara.display(task.value['event'])
            kwds=task.value['kwds']
            def switch(kde, co):
                kde[co] = not kde[co]
                return kde
            solara.Text("Switchers: ")
            solara.Button(label = "Full time/Just pulling", on_click=lambda : task(task.value['event'],  kwds=switch(kwds,'ctrl')))
            solara.Button(label = "x axis: Force/Time", on_click=lambda : task(task.value['event'], kwds=switch(kwds,'alt')))
            solara.Button(label = "Points/Lines", on_click=lambda : task(task.value['event'], kwds=switch(kwds,'shift')))


def click_figure(data,event, shift=False, ctrl=False, alt=False):
    tree, mt, pullNo, indep, dep, measurement, camera, R, day = list(data)
    if shift == True:
        mode = 'markers'
    else:
        mode = 'lines'
    show_dialog.value = True
    if "_" in str(pullNo):
        pullNo = int(str(pullNo).split("_")[-1])
    if alt == True:
        restricted = [.3,.9]
    else:
        restricted = None
    m = static_pull.DynatreeStaticMeasurement(day=day, tree=tree, measurement=measurement, measurement_type=mt,
                                              optics=False, restricted=restricted)
    if indep in ["blueMaj", "yellowMaj"]:
        subtitle = f"{indep}, {m.identify_major_minor[indep]}"
    else:
        subtitle = indep
    if ctrl == False:
        pull = m.pullings[pullNo]
        df = pull.data
    else:
        if indep in ["blueMaj", "yellowMaj"]:
            indep = m.identify_major_minor[indep]
        else:
            indep = "Elasto(90)"
        df = m.data_pulling[[indep, "Force(100)"]]


    # rich.print(df.columns)
    if alt == True:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df[["Force(100)"]].to_numpy().reshape(-1), x=df[[indep]].to_numpy().reshape(-1), mode='markers'))
        fig.update_layout(
            yaxis_title="Force",  # Nastavení popisku osy X
            xaxis_title=indep,
        )
    else:
        fig = make_subplots(rows=2, cols=1, subplot_titles=(subtitle, "Force"), shared_xaxes=True)
        fig.add_trace(go.Scatter(x=df.index.to_list(), y=df[[indep]].to_numpy().reshape(-1), mode=mode), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index.to_list(), y=df[["Force(100)"]].to_numpy().reshape(-1), mode=mode), row=2, col=1)
        fig.update_layout(hovermode = "x unified")
        fig.update_traces(xaxis='x2')
        fig.update_layout(
            xaxis2_title="Time",  # Nastavení popisku osy X
            yaxis_title=indep,
            yaxis2_title="Force"
        )
    newdata = [str(data[i]) for i in [1,-1,0,5, 2,3,4,6, 7]]
    newdata[-2] = "Kamera "+newdata[-2]
    newdata[4] = "pullNo "+newdata[4]
    fig.update_layout(title=", ".join(newdata), height=s.height.value)
    return fig

figdata = None
@solara.component
def slope_trend():
    global figdata

    with solara.Sidebar():
        limitR2()
        limit_statics_dynamics()
    df = static_lib_pull_comparison.df_all_M
    df = df[(df["R^2"]>=R2limit_lower.value) & (df["R^2"]<=R2limit_upper.value)]
    if not include_statics.value:
        df = df[df['measurement'] != "M01"]
    if not include_dynamics.value:
        df = df[df['measurement'] == "M01"]
    independent = probe.value
    if independent == "Elasto-strain":
        filtered_df = df[df['Independent'] == independent]
    elif independent == "Camera":
        filtered_df = df[df['kamera'] == True]
    else:
        filtered_df = df[df['kamera'] == False]
    filtered_df = filtered_df[filtered_df["tree"] == s.tree.value]
    subdf = filtered_df.sort_values(by="day")
    cat_order = subdf["day"].drop_duplicates().tolist()
    filtered_df["kamera"] = filtered_df["kamera"].astype(str)
    filtered_df["Day"] = filtered_df["day"]
    # Vykreslení boxplotu
    fig = plx.box(
        filtered_df,
        x='day',  # Kategorická osa X
        y='Slope',  # Hodnota pro osy Y
        color='type',  # Barvení podle sloupce 'type'
        title=f'Slope by Day and Type, tree {s.tree.value}, slope for momentum and {probe.value}',
        category_orders={"day": cat_order},
        template="plotly_white",
        hover_data=["tree", "type", "day", "pullNo", "Independent", "Dependent", "measurement","kamera", "R^2", "Day"],
        points='all',
        width=s.width.value,
        height=s.height.value,
        color_discrete_map= cmap
        # box=True,
        # symbol='measurement',      # Tvar bodů na základě sloupce 'measurement'
    )
    fig.update_layout(xaxis=dict(type='category'))
    click_info()
    solara.FigurePlotly(fig, on_click = on_click)
    figdata = fig
    overlay(on_click)
    solara.FileDownload(filtered_df.to_csv(), filename="tahovky_trend_I.csv")
    # solara.DataFrame(filtered_df.sort_values(by="Slope"))
    great_table = (
        GT(filtered_df[["type", "day", "tree", "measurement", "pullNo", "Slope"]]
           .sort_values(by="Slope"))
        .fmt_scientific("Slope")
        .tab_style(
            style=[
                style.fill(color="lightblue"),
                style.text(weight="bold")
            ],
            locations=loc.body(columns="Slope")
        )
        .tab_header(title=f"Slope of momentum versus {independent}")
        .tab_spanner(label="Measurement", columns=["type", "day", "tree", "measurement", "pullNo"])
        .cols_label({"type": "", "day": "", "measurement": "", "tree": "", "pullNo": ""})
    )
    # .fmt_nanoplot("Slope", plot_type="bar")
    solara.display(great_table)
    # solara.FileDownload(HTML(string=great_table.as_raw_html()).write_pdf(), filename=f"dynatree-tahovky-table.pdf", label="Download PDF")


color = solara.reactive("pullNo")


# Funkce pro stylování - přidání hranice, když se změní hodnota v úrovni 'tree'
def add_vertical_line(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)

    # Projdi všechny řádky a přidej stylování
    for i in range(1, len(df.columns)):
        if df.columns[i][0] != df.columns[i - 1][0]:
            styles.iloc[:, i] = 'border-left: 5px solid lightgray'  # Přidej hranici
    return styles


# Funkce pro detekci změn v první úrovni MultiIndexu
def highlight_changes(col):
    first_level = col.columns.get_level_values(0)
    # Porovnáme s předchozí hodnotou, kde je změna, tam vrátíme styl
    return ['border-right: 3px solid black' if first_level[i] != first_level[i - 1] else '' for i in
            range(len(first_level))]


def ostyluj(subdf, skip_last_column=False, skip_how_many=1):
    cm = sns.light_palette("blue", as_cmap=True)
    columns_to_style = subdf.columns
    if skip_last_column:
        columns_to_style = columns_to_style[:-skip_how_many]
    vmin = subdf.loc[:,columns_to_style].min(skipna=True).min()
    subdf = (subdf.style.format(precision=3)
             .background_gradient(vmin=vmin, axis=None, subset=columns_to_style)
             .format(na_rep='')
             .apply(add_vertical_line, axis=None, subset=columns_to_style)
             # .apply(highlight_changes, axis=0, subset=pd.IndexSlice[:, :])
             .map(lambda x: 'color: lightgray' if pd.isnull(x) else '', subset=columns_to_style)
             .map(lambda x: 'background: transparent' if pd.isnull(x) else '', subset=columns_to_style)
             )
    return subdf


@solara.component
def slope_trend_more():
    global figdata

    with solara.Sidebar():
        limitR2()
    with solara.Row():
        with solara.Tooltip(solara.Markdown(
                """
                * Můžeš vybrat "pullNo" (číslo zatažení) a sledovat, jestli tečky jiných barev
                  vykazují nějaký trend, například jestli je tečka pro nulté zatažení stabilně pod 
                  nebo nad tečkou pro další zatažení. V tomto případě se pracuje jenom s M01.
                * Můžeš vybrat "kamera" a sledovat časový vývoj inklinometru v dané pozici stromu.
                """, style={'color': 'white'})):
            solara.Text("Barevně separovat podle ⓘ:")
            solara.ToggleButtonsSingle(value=color, values=["pullNo", "kamera", "category"])
    df = (pd.read_csv("../outputs/anotated_regressions_static.csv", index_col=0)
          .pipe(lambda x: x[x['lower_cut'] == 0.3])
          .pipe(lambda x: x[x['tree'] == s.tree.value])
          # .pipe(lambda x: x[x['measurement'] == 'M01'])
          .pipe(lambda x: x[~x['failed']])
          .pipe(lambda x: x[~x['optics']])
          # .pipe(lambda x: x[~(x['Dependent'].str.contains('Min'))])
          .pipe(lambda x: x[x['tree'].str.contains('BK')])
          .pipe(lambda x: x[x['Dependent'] == "M"])
          .pipe(lambda x: x[["type", "day", "tree", "Independent", "Dependent", "Slope", "pullNo", "measurement", "kamera", "R^2"]])
          # .pivot(values="Slope", columns='pullNo', index=
          #        ['type', 'day', 'tree', 'measurement', 'Dependent'])
          )
    df = df[(df["R^2"]>=R2limit_lower.value) & (df["R^2"]<=R2limit_upper.value)]
    df["pullNo"] = df["measurement"].astype(str) + "_" + df["pullNo"].astype(str)
    df["category"] = "dynamics"
    df.loc[df["measurement"]=="M01", "category"] = "statics"
    if color.value == "pullNo":
        df = df.pipe(lambda x: x[x['measurement'] == 'M01'])
    # breakpoint()
    df["Slope / 1000"] = df["Slope"] / 1000
    df["Slope / 1000"] = df["Slope"] / 1000
    df["id"] = df["day"] + " " + df["type"]
    fig = plx.strip(df, x="id", y="Slope", template="plotly_white",
                    color=color.value,
                    hover_data=["tree", "type", "pullNo", "Independent", "Dependent", "measurement", "kamera", "R^2", "day"],
                    title=f"Tree {s.tree.value}, inclinometers, slope from the angle-momentum relationship.",
                    width=s.width.value, height=s.height.value
                    )
    figdata = fig
    click_info()
    overlay(on_click)
    solara.FigurePlotly(fig, on_click=on_click)
    with solara.Info():
        solara.Markdown(
            """
            * V tabulce jsou data pro M01, pokud je vybráno "PullNo" a všechna data, pokud je vybráno "kamera" nebo "category".
            * Barvné rozseparování podle pullNo (číslo zatáhnutí) umožní sledovat, jestli 
              se během experimentu liší první zatáhnutí od ostatních a jak. 
            * Barevné rozseparování podle kamera umožní studovat časový vývoj v daném místě stromu. Kamera true/false 
              rozlišuje, zda je přísroj vidět na boční kameře.
            """,
            style={'color': 'inherit'}
        )
    df = df.pivot(index=["day", "type"], columns=["kamera", "pullNo"], values="Slope")
    df_kamera = static_pull.DF_PT_NOTES["kamera"].copy().reset_index()
    df_kamera = df_kamera[df_kamera["tree"] == s.tree.value].drop("tree", axis=1).set_index(["day","type"])
    df_kamera.columns = [("kamera","")]
    df[("kamera","")] = np.nan
    df[("kamera", "")] = df[("kamera", "")].astype(object)
    df.update(df_kamera)
    df = df.sort_index(axis=1)
    solara.display(ostyluj(df, skip_last_column=True))
    solara.Style(
        ".col_heading.level0 {text-align: center !important; background-color: lightgray; border-right: 5px solid white !important;}")
    solara.FileDownload(df.to_csv(), filename=f"tahovky_{s.tree.value}.csv",
                        label="Download data from this table")
    return


@solara.component
def normalized_slope():
    df_merged = static_lib_pull_comparison.df_merged
    subdf = df_merged[df_merged["pullNo"] != 0].loc[:,
            ["type", "day", "tree", "Independent", "kamera", "pullNo", "Slope_normalized"]]
    subdf = subdf[subdf["tree"] == s.tree.value].sort_values(by="day")
    cat_order = subdf["day"].drop_duplicates().tolist()
    subdf["kamera"] = subdf["kamera"].astype(str)
    fig = plx.box(
        subdf,
        x="day",
        y="Slope_normalized",
        color='type',
        points='all',
        hover_data=["tree", "type", "pullNo", "Independent", "kamera"],
        category_orders={"day": cat_order},
        height=s.height.value, width=s.width.value,
        title=f"Tree {s.tree.value}",
        template="plotly_white",
        color_discrete_map= cmap
    )
    fig.update_layout(xaxis=dict(type='category'))
    solara.FigurePlotly(fig)
    #     solara.Text(
    # """
    # V tabulce jsou data seřazená podle normalizovnaé směrnice.
    # Kliknutím na buňku se zobrazí link, který nahraje meření a přístroj do
    # vedlejší záložky 'Volba proměnných a regrese'. Automaticky se zobrazí
    # časový průběh, možná budeš chtít zatrhnout 'Ignore time restriction',
    # aby se zobrazil celý pokus a ne jenom natahování.
    # """)
    solara.Switch(label="řadit od nejmenšího", value=sort_ascending)
    subdf = subdf.sort_values(by="Slope_normalized", ascending=sort_ascending.value)
    solara.DataFrame(
        subdf,
        items_per_page=20,
        # cell_actions=cell_actions
    )
    solara.FileDownload(subdf.to_csv(), filename="normalized_slope_static.csv")

probe = solara.reactive("Elasto-strain")
probes = ["Elasto-strain", "Camera", "NoCamera"]
how_to_colorize = solara.reactive("All data")
restrict_to_noc_den = solara.reactive(False)
include_M01 = solara.reactive(True)


def mysort(df):
    type_order = ['normal', 'noc', 'den', 'afterro', 'afterro2', 'mraz', 'mokro']
    df['type'] = pd.Categorical(df['type'], categories=type_order, ordered=True)
    df = df.sort_values(["tree", "day", "type"])
    return df
    # if restrict_type is not None:
    #     mask = df['type'].isin(restrict_type)
    #     df = df[mask]

def custom_display(df_, all_data=True, second_level=False):
    with solara.Row():
        solara.Switch(label="restrict to den/noc", value=restrict_to_noc_den)
        solara.Switch(label="include M01", value=include_M01)
    df = df_.copy()
    index_names = df.index.names
    df = df.reset_index().pipe(mysort)
    if restrict_to_noc_den.value:
        mask = df["type"].isin(["den","noc"])
        df = df[mask]
    if not include_M01.value:
        mask = [i for i in df.columns if "M01" not in i[1]]
        df = df.loc[:,mask]
    df = df.set_index(index_names)
    if all_data:
        solara.display(du.ostyluj(df, second_level=second_level))
    else:
        for category in df.index.get_level_values(0).unique():
            # Získáme řádky odpovídající dané skupině
            group_df = df.xs(category, level=0)
            with solara.Card(margin=4):
                solara.Markdown(f"##{category}")
                # Zobrazíme tabulku s aplikovaným gradientem pro tuto skupinu
                solara.display(ostyluj(group_df))

@solara.component
def limitR2():
    with solara.Card(title="Limit for R^2"):
        solara.InputFloat("lower bound", value=R2limit_lower)
        solara.InputFloat("upper bound", value=R2limit_upper)

def limit_statics_dynamics():
    with solara.Card(title="Include statics/dynamics?"):
        solara.Checkbox(label="Statics (M01)", value=include_statics)
        solara.Checkbox(label="Dynamics (M02, M03, ...)", value=include_dynamics)

@solara.component
def Page():
    solara.Title(title)
    solara.Style(s.styles_css)
    with solara.Sidebar():
        if tab_index.value in [1]:
            s.Selection_trees_only()
        if tab_index.value == 0:
            Selection()
        if tab_index.value == 3:
            solara.Markdown(
                """
                * Na této záložce jsou ke stažení csv soubory, které řídí výpočet. 
                * Výsledky jsou ke stažení na 
                stránce Downloads.
                """
            )

        if tab_index.value in [0, 1]:
            s.ImageSizes()
            s.width.value = 1200
    dark = {"background_color": "primary", "dark": True, "grow": True}
    with solara.lab.Tabs(value=tab_index, **dark):
        with solara.lab.Tab("Jedno měření (detail, ...)", icon_name="mdi-chart-line"):
            with solara.lab.Tabs(lazy=True, value=subtab_index, **dark):
                with solara.lab.Tab("Regrese pro získání tuhostí"):
                    if (tab_index.value, subtab_index.value) == (0, 0):
                        dynatree.logger.info("Regrese pro ziskani tuhosti")
                        with solara.Card(title="Regressions to get the stiffness"):
                            # Detail()
                            try:
                                Regrese()
                            except:
                                solara.Error("Něco se pokazilo při volání funkce Regrese...")
                with solara.lab.Tab("Průběh síly"):
                    if (tab_index.value, subtab_index.value) == (0, 1):
                        dynatree.logger.info("Zakladni graf")
                        with solara.Card():
                            Graphs()
                with solara.lab.Tab("Detaily závislostí"):
                    if (tab_index.value, subtab_index.value) == (0, 2):
                        dynatree.logger.info("Volba promennych a diagramy")
                        with solara.Card(title="Increasing part of the time-force diagram"):
                            # Detail()
                            try:
                                Detail()
                            except:
                                solara.Error("Něco se pokazilo při volání funkce Detail...")
                with solara.lab.Tab("Polární graf"):
                    if (tab_index.value, subtab_index.value) == (0, 3):
                        dynatree.logger.info("Polarni graf")
                        with solara.Card():
                            try:
                                Polarni()
                            except:
                                solara.Error("Něco se pokazilo ...")

        with solara.lab.Tab("Jeden strom (trend, ...)", icon_name="mdi-pine-tree"):
            with solara.lab.Tabs(lazy=True, **dark):
                with solara.lab.Tab("Srovnání s prvním zatáhnutím"):
                    with solara.Card():
                        solara.Markdown(
                            """
                            **Srovnání následujících zatáhnutí s prvním**
                            
                            * V grafech je podíl směrnice z druhého nebo třetího zatáhnutí a směrnice z prvního zatáhnutí. Toto je v grafu vedeno jako Slope_normalized.
                            * Pokud věříme, že při prvním zatáhnutí je systém tužší, měl by podíl být stabilně pod jedničkou.
                            * V sidebaru vlevo můžeš přepínat strom, graf by se měl automaticky aktualizovat.
                            """)
                        try:
                            normalized_slope()
                        except:
                            solara.Error("Něco se pokazilo ...")

                            # solara.FigurePlotly(figPl)                
                with solara.lab.Tab("Hledání odlehlých"):
                    with solara.Column():
                        try:
                            prehled()
                        except:
                            solara.Error("Něco se pokazilo ...")
                with solara.lab.Tab("Trend (1 senzor)"):
                    with solara.Sidebar():
                        if tab_index.value == 1:
                            with solara.Card():
                                solara.Markdown("**Variable** is elastometer or one of inclinometers (Camera and NoCamera). ")
                                solara.ToggleButtonsSingle(value=probe, values=probes, on_value=slope_trend)
                    with solara.Column():
                        slope_trend()
                with solara.lab.Tab("Trend (oba inklinometry)"):
                    with solara.Column():
                        slope_trend_more()
        with solara.lab.Tab("Všechny stromy", icon_name="mdi-file-table-box-multiple-outline"):
            if tab_index.value == 2:
                with solara.Sidebar():
                    solara.Markdown("**Gradient**")
                    solara.ToggleButtonsSingle(value=how_to_colorize, values=["All data", "Within tree"])
                    limitR2()
                    solara.Markdown("**Návod:**")
                    solara.Markdown(
                        """
                        * **All data**: jako rozsah se berou všechna data. Tužší stromy jsou jinou barvou než poddajnější. 
                          Dobré pro kontrolu, jestli v rámci stromu jsou data plus minus stejná.
                        * **Within tree**: jako rozsah se berou data pro daný strom. Slouží k nalezení měření, kdy strom 
                          byl tužší nebo poddajnější než obvykle a ve srovnání se všemi daty by tento rozdíl zapadl.
                        * **Limit for R^2:**: Nechat v tabulce jenom regrese, kde je R^2 v povolených mezích.
                        """
                    )
                with solara.lab.Tabs(lazy=True, **dark):
                    with solara.lab.Tab("Camera"):
                        show_regression_data_inclino("Camera")
                    with solara.lab.Tab("NoCamera"):
                        show_regression_data_inclino("NoCamera")
                    with solara.lab.Tab("Elasto strain"):
                        show_regression_data_elasto()
                    with solara.lab.Tab("Pt3"):
                        show_regression_data_pt("Pt3")
                    with solara.lab.Tab("Pt4"):
                        show_regression_data_pt("Pt4")
        with solara.lab.Tab("Komentáře & dwnl.", icon_name="mdi-comment-outline"):
            with solara.Card(title="Návod"):
                Help()


def read_regression_data():
    df = pd.read_csv("../outputs/anotated_regressions_static.csv", index_col=0)
    df["M"] = df["measurement"]
    mask = df["measurement"] == "M01"
    df.loc[mask, "M"] = df.loc[mask, "measurement"] + "_" + df.loc[mask, "pullNo"].astype("str")
    df = df[df["tree"].str.contains("BK")]
    df = df[(df["lower_cut"] == 0.3) & (~df["failed"])]
    return df

R2limit_upper = solara.reactive(1)
R2limit_lower = solara.reactive(0.9)

@solara.component
def show_regression_data_inclino(color, restrict_type=["den","noc"], include_M01=True):
    df = read_regression_data()
    if color == "Camera":
        df = df[df["kamera"] == True]
        df = df[df["Dependent"] == "M"]
    elif color == "NoCamera":
        df = df[df["kamera"] == False]
        df = df[df["Dependent"] == "M"]
    else:
        df = df[df["Independent"] == color]
    df["Slope x 1e-3"] = 1e-3 * df["Slope"]
    df = df[~df["optics"]]
    df = df[df["R^2"]>=R2limit_lower.value]
    df = df[df["R^2"]<=R2limit_upper.value]
    df_final = df.pivot(index=["tree", "day", "type"], values=["Slope x 1e-3"], columns="M")
    custom_display(df_final, how_to_colorize.value == "All data", second_level=True)


@solara.component
def show_regression_data_elasto():
    df = read_regression_data()
    df = df[df["Independent"] == "Elasto-strain"]
    df = df[~df["optics"]]
    df["Slope x 1e-6"] = 1e-6 * df["Slope"]
    df = df[df["R^2"]>=R2limit_lower.value]
    df = df[df["R^2"]<=R2limit_upper.value]
    df_final = df.pivot(index=["tree", "type", "day"], values=["Slope x 1e-6"], columns="M")
    custom_display(df_final, how_to_colorize.value == "All data", second_level=True)


@solara.component
def show_regression_data_pt(pt):
    df = read_regression_data()
    df = df[df["Independent"] == pt]
    df["Slope"] = np.abs(df["Slope"])
    df = df[df["R^2"]>=R2limit_lower.value]
    df = df[df["R^2"]<=R2limit_upper.value]
    df_final = df.pivot(index=["tree", "type", "day"], values=["Slope"], columns="M")
    custom_display(df_final, how_to_colorize.value == "All data", second_level=True)


@solara.component
def Selection():
    s.Selection()
    data_object = dynatree.DynatreeMeasurement(
        s.day.value, s.tree.value, s.measurement.value, measurement_type=s.method.value)
    with solara.Column(align='center'):
        solara.Button("Run calculation", on_click=nakresli, color="primary")


def fixdf(df):
    df.columns = [f"{i[0]}" if i[1] == 'nan' else f"{i[0]}_{i[1]}" for i in df.columns]
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
    with solara.Columns([6, 6]):
        for df, title in zip(l, titles):
            with solara.Card():
                solara.Markdown(f"**{title}**")
                df = df[[i for i in df.columns if "fixed" not in i[0]]]
                nans = pd.DataFrame(df.isna().sum())
                nans.loc[:, "name"] = df.columns
                nans.columns = ["#nan", "name"]
                nans = nans[["name", "#nan"]]
                solara.Markdown(f"Shape: {df.shape}")
                solara.DataFrame(nans)
    # try:
    #     solara.DataFrame(pd.concat([pd.DataFrame(subdf.index),subdf], axis=1))
    # except:
    #     pass


@solara.component
def Graphs():
    dynatree.logger.info("Function Graph entered")
    solara.ProgressLinear(nakresli.pending)
    if s.measurement.value not in available_measurements(s.df.value, s.day.value, s.tree.value, s.method.value):
        with solara.Error():
            solara.Markdown(
                f"""
                * Measurement {s.measurement.value} not available for tree {s.tree.value}
                  day {s.day.value} measurement type {s.method.value}.
                * You may need to switch measurement type (normal/den/noc/...) 
                  if the list of the measuemrent day is incorrect.
                """, style={'color':'inherit'})
        return

    if nakresli.not_called:
        solara.Info(solara.Markdown("""
                        
* Vyber měření a stiskni tlačítko \"Run calculation\". 
* Ovládací prvky jsou v sidebaru. Pokud není otevřený, otevři kliknutím na tři čárky nalevo v modrém pásu.
* Při změně vstupů se většinou obrázek aktualizuje, ale ne vždy. Pokud nadpis na obrázku nesouhlasí s vybranými hodnotami, spusť výpočet tlačítkem \"Run calculation\".

                                    """, style={'color':'inherit'}))
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
        solara.Markdown(
            "Na obrázku je průběh experimentu (časový průběh síly). Volitelně můžeš zobrazit detaily pro rozmezí 30%-90% maxima síly. \n\n V detailech je časový průběh síly, časový průběh na inklinometrech a grafy inklinometry versus síla nebo moment.")
        with solara.Tooltip("Allows to show details of the pulls on this page. Slows down the computation, however."):
            solara.Switch(label="Show details", value=include_details, on_value=nakresli)
        f = nakresli.value
        if f is None:
            solara.Error("Obrázek/obrázky se nepodařilo vytvořit.")
            return
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
    dynatree.logger.info("Function Polarni entered")
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
                    pulls = list(range(len(temp_data_object.pullings)))+["All"]
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
    if pull_value == "All":
        pull_list = pulls
    else:
        pull_list = [pull_value]
    fig, ax = plt.subplots()
    for i,one_pull in enumerate(pull_list):
        if one_pull == "All":
            continue
        dataset = d_obj.pullings[one_pull]
        subdf = dataset.data
        ax.plot(subdf['blueMaj'], subdf['blueMin'], label=f"blue{i}")
        ax.plot(subdf['yellowMaj'], subdf['yellowMin'], label=f"yellow{i}")
    ax.set_aspect('equal')
    ax.legend()
    bound = [*ax.get_xlim(), *ax.get_ylim()]
    bound = np.abs(np.array(bound)).max()
    title = f"{d_obj.measurement_type} {d_obj.day} {d_obj.tree} {d_obj.measurement}"
    if d_obj.measurement == "M01":
        title = f"{title}, PullNo {pull_value}"
    ax.set(xlim=(-bound, bound), ylim=(-bound, bound), title=title, xlabel="Major inclinometer",
           ylabel="Minor inclinometer")
    ax.grid(which='both')
    solara.FigureMatplotlib(fig)
    plt.close('all')


def Regrese():
    dynatree.logger.info("Function Regrese entered")
    with solara.Info():
        solara.Markdown("""
        * Obrázky pro posouzení dat, ze kterých se počítají tuhosti.
        * Pokud chceš vidět, jaká konkrétní data tu jsou a zobrazovat si je různými způsoby 
          (časový průběh, volba veličin na osách, volba ořezu atd), použij třetí podzáložku 
          (Jedno měření, detaily závislostí).
        """, style={'color':'inherit'})
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
        if (s.use_optics.value) and (not temp_data_object.is_optics_available):
            s.use_optics.value = False
            return

    restricted = (0.3, 0.9)

    d_obj = static_pull.DynatreeStaticMeasurement(
        day=s.day.value, tree=s.tree.value,
        measurement=s.measurement.value,
        measurement_type=s.method.value,
        optics=s.use_optics.value,
        restricted=restricted)
    dataset = d_obj.pullings[pull_value]
    subdf = dataset.data

    title = f"{s.day.value} {s.tree.value} {s.measurement.value} {s.method.value} Pull {pull_value}"

    fig, ax = plt.subplots()
    subdf.plot(x="blueMaj", y="M", style='.', ax=ax, legend=False)
    subdf.plot(x="yellowMaj", y="M", style='.', ax=ax, legend=False)
    ax.legend(["blueMaj","yellowMaj"])
    # ax.set(ylim=(subdf[ydata.value].to_numpy().min(), subdf[ydata.value].to_numpy().max()))
    ax.grid()
    ax.set(title=title, xlabel="Inclinometers", ylabel="M")

    fig2, ax2 = plt.subplots()
    subdf.plot(x="Elasto-strain", y="M_Elasto", style='.', ax=ax2, legend=False)
    # ax.set(ylim=(subdf[ydata.value].to_numpy().min(), subdf[ydata.value].to_numpy().max()))
    ax2.grid()
    ax2.set(title=title, xlabel="Elasto-strain", ylabel="M_Elasto")
    with solara.Card(
            style={"max-width": "1800px"}
    ):
        with solara.Row():
            solara.FigureMatplotlib(fig)
            solara.FigureMatplotlib(fig2)
    plt.close('all')


def Detail():
    dynatree.logger.info("Function Detail entered")
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
    with solara.Info():
        solara.Markdown("""
            * **Pokud chceš vidět tuhost, dej na vodorovnou osu náklon a na svislou osu moment.** 
            * Pro výběr proměnných na vodorovnou a svislou osu otevři menu v sidebaru (tři čárky v horním panelu). Po výběru můžeš sidebar zavřít. Přednastavený je moment vypočítaný z pevného naměřeného úhlu lana na vodorovné ose a oba inlinometry na svislé ose, 
            aby šly vidět oba inlikonmetry na stejném definičním oboru. Pro směrnici udávající moment to potřebuješ přehodit nebo jít na první záložku. 
            """, style={'color':'inherit'})

    with solara.Sidebar():
        cols = ['Time', 'Pt3', 'Pt4', 'Force(100)', 'Elasto(90)', 'Elasto-strain',
                # 'Inclino(80)X', 'Inclino(80)Y', 'Inclino(81)X', 'Inclino(81)Y',
                # 'blue', 'yellow',
                'blueMaj', 'yellowMaj',
                'F_horizontal', 'F_vertical',
                'M', 'M_Pt', 'M_Elasto',
                ]
        with solara.Card(title="Horizontal axis"):
            solara.Markdown("Choose one variable.")
            solara.ToggleButtonsSingle(values=cols, value=xdata, dense=True)
        with solara.Card(title="Vertical axis"):
            solara.Markdown(
                "Choose one or more variables. You cannot choose the same variable which has been used for horizontal axis. You cannot choose optical probes if the 'Use optics' switch is off.")
            solara.ToggleButtonsMultiple(
                values=cols[1:], value=ydata, dense=True)
        with solara.Card(title="Second vertical axis"):
            with solara.VBox():
                with solara.Tooltip(
                        "Choose one variable for second vertical axis, shown on the right. (Only limited support in interactive plots. In interactive plots we plot rescaled data. The scale factor is determined from maxima.) You cannot choose the variable used for horizontal axis."):
                    with solara.VBox():
                        solara.Text("🛈 (hover here for description)")

                solara.ToggleButtonsSingle(
                    values=[None] + cols[1:], value=ydata2, dense=True)
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
                with solara.Tooltip(
                        "Umožní zobrazit graf pomocí knihovny Plotly. Bude možné zoomovat, odečítat hodnoty, klikáním na legendu skrývat a odkrývat proměnné apod. Nebudou zobrazeny regresní prímky."):
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
        parent_experiment = static_pull.DynatreeStaticMeasurement(
            tree=s.tree.value, measurement_type=s.method.value,
            day=s.day.value, measurement=s.measurement.value,
        )
        subdf = static_pull.DynatreeStaticPulling(_, tree=s.tree.value, measurement_type=s.method.value,
                                                  day = s.day.value,
                                                  parent_experiment=parent_experiment,
                                                  extra_columns={"blue": "Inclino(80)", "yellow": "Inclino(81)",
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
            reg_df = static_pull.DynatreeStaticPulling._get_regressions(subdf, [[xdata.value] + target],
                            coords = (s.method.value, s.day.value, s.tree.value, s.measurement.value))
            solara.DataFrame(reg_df.iloc[:, :5])
            # solara.display(reg_df.iloc[:, :5])
            df_subj_reg = subdf[[xdata.value] + target]
        else:
            solara.Info(
                """
                No regressions are calculated if the independent variable is Time or if all
                data are considered. (Put "Ignore time restriction off and select another independent variable.")
                """)
    except:
        solara.Error(
            "Něco se (možná) pokazilo při hledání regresí. Nahlaš prosím problém. Pro další práci vyber jiné veličiny. Pokud tato hláška během chvíle zmizí, je neškodná.")

    title = f"{s.day.value} {s.tree.value} {s.measurement.value} {s.method.value} Pull {pull_value}"
    if interactive_graph.value:
        kwds = {"template": "plotly_white", "height": 600, "title": title}
        try:
            if ydata2.value != None:  # Try to add rescaled column
                maximum_target = np.nanmax(
                    subdf[fix_input(ydata.value)].values)
                maximum_ori = np.nanmax(subdf[ydata2.value].values)
                subdf.loc[:, f"{ydata2.value}_rescaled"] = subdf.loc[:,
                                                           ydata2.value] / np.abs(maximum_ori / maximum_target)
                extradata = [f"{ydata2.value}_rescaled"]
            else:
                extradata = []
            cols_to_draw = fix_input(ydata.value + extradata)
            if xdata.value == "Time":
                fig = plx.scatter(subdf, y=cols_to_draw, **kwds)
            else:
                subdf["Time"] = subdf.index
                fig = plx.scatter(subdf, x=xdata.value, y=cols_to_draw, hover_data=["Time"],
                                  **kwds)
            solara.FigurePlotly(fig)
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
                    ax.plot(t, t * d.iat[0, 0] + d.iat[0, 1],
                            **regression_settings)
        except:
            pass
        ax.set(ylim=(subdf[ydata.value].to_numpy().min(), subdf[ydata.value].to_numpy().max()))
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
            solara.FileDownload(df_subj_reg.to_csv(),
                                filename=f"regression_{s.method.value}_{s.day.value}_{s.tree.value}_{s.measurement.value}.csv",
                                label="Download as csv")
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
    with solara.Card(title="Použité parametry"):
        solara.Markdown(""" 
        * [`csv/static_fail.csv`](https://github.com/arborist-mendelu/dynatree/blob/master/scripts/csv/static_fail.csv) 
          Zkoušky klasifikované jako nepovedené
        * [`ccsv/static_checked_OK.csv`](https://github.com/arborist-mendelu/dynatree/blob/master/scripts/csv/static_checked_OK.csv) 
          Zkoušky klasifikované jako OK, i když se hodnoty liší od ostatních
        * [`csv/reset_inclinometers.csv`](https://github.com/arborist-mendelu/dynatree/blob/master/scripts/csv/reset_inclinometers.csv) 
          Ručně vynulované inklinometry
        * [`csv/static_manual_limits.csv`](https://github.com/arborist-mendelu/dynatree/blob/master/scripts/csv/static_manual_limits.csv) 
          Ručně nastavené časové limity pro regresi. Používá se, pokud regrese není pěkná, ale dá se identifikovat, že k problému došlo na začátku nebo na konci a zbytek 
          je pro regresi dostetačně dlouhý.
        """, style={'color':'inherit'})

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

dynatree.logger.info(f"File tahovky.py loaded in {time.time()-loading_start} sec.")

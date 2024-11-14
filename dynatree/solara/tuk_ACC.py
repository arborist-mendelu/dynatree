from matplotlib.pyplot import axvline
import solara.lab
import solara.website
from dynatree import dynatree
import solara
import pandas as pd
import polars as pl
import dynatree.solara.select_source as s
import matplotlib.pyplot as plt
from solara.lab import task
from dynatree.signal_knock import SignalTuk, find_peak_times_chanelA, find_peak_times_chanelB, chanelA, chanelB
from dynatree_summary.acc_knocks import  delta_time
import logging
import plotly.express as px

from dynatree.solara.tahovky import interactive_graph
import numpy as np
from contextlib import contextmanager

# dynatree.logger.setLevel(logging.INFO)
dynatree.logger.setLevel(logging.INFO)
df = pd.read_csv("../outputs/FFT_acc_knock.csv")
df["valid"] = True
df["manual_peak"] = np.nan
rdf =  solara.reactive(df)

active_tab = solara.reactive(1)
use_overlay = solara.reactive(False)

@contextmanager
def customizable_card(overlay=True):
    if overlay:
        # První varianta s overlay efektem přes celou obrazovku
        with solara.Card(style={
            'position': 'fixed', 'top': '0', 'left': '0', 'width': '100vw', 'height': '100vh',
            'background-color': 'rgba(0, 0, 0, 0.5)', 'z-index': '999',
            'display': 'flex', 'align-items': 'center', 'justify-content': 'center'
        }, margin=0):
            with solara.Card(style={
                'background-color': 'white', 'padding': '20px', 'border-radius': '8px',
                'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.2)',
                'max-width': '1200px', 'width': '1000px', 'max-height': '100vh'
            }):
                yield  # Výkon příkazu v této variantě
    else:
        # Druhá varianta, fixovaná v pravém dolním rohu
        with solara.Card(style={
            'position': 'fixed', 'bottom': '0px', 'right': '0px', 'z-index': '1000',
            'max-width': '1200px', 'width': '1000px', 'max-height': '100vh'
        }):
            yield  # Výkon příkazu v této variantě


@solara.component
def Page():
    dynatree.logger.info("Page entered")
    solara.Title("DYNATREE: ACC ťuknutí")
    solara.Style(s.styles_css)
    with solara.Sidebar():
        s.Selection(
            optics_switch=False,
            report_optics_availability=False,
            include_measurements=active_tab.value != 2
        )
        if active_tab.value == 1:
            solara.Switch(label="Use overlay for graphs", value=use_overlay)
            solara.FileDownload(rdf.value.to_parquet(), filename="FFT_acc_knock.parquet",
                                label="Download data")
    with solara.lab.Tabs(value=active_tab, lazy=True):
        with solara.lab.Tab("Celé měření"):
            Signal()
            Rozklad()
        # with solara.lab.Tab("Tabulka"):
        #     Tabulka()
        with solara.lab.Tab("Detail"):
            if use_overlay.value:
                Seznam()
                Graf()
            else:
                with solara.Columns(1,1):
                    with solara.Column():
                        Seznam()
                    with solara.Column():
                        Graf()
        with solara.lab.Tab("Tabulková forma"):
            Tabulka()



@solara.component
def Graf():
    if interactive_graph.finished:
        ans = interactive_graph.value
        if ans is None:
            return
    if interactive_graph.not_called:
        return
    with customizable_card(use_overlay.value):
        solara.ProgressLinear(interactive_graph.pending)
        if interactive_graph.finished:
            ans = interactive_graph.value
            with solara.Row():
                solara.Text(ans['text'])
                solara.Text(f"(id {ans['index']})")

                solara.Button("❌", on_click=lambda: interactive_graph())
            solara.Markdown("**Časový průběh**")
            solara.FigurePlotly(ans['signal'])
            solara.Markdown(f"**FFT transformace** Peak at Hz.")
            solara.FigurePlotly(ans['fft'])

@task
def interactive_graph(type=None, day=None, tree=None, measurement=None, probe=None, start=None, index=None):
    if type is None:
        return None
    dynatree.logger.info(f"interactive graph entered {day} {tree} {measurement} {type}")
    mi = dynatree.DynatreeMeasurement(day=day, tree=tree, measurement=measurement,
                                     measurement_type=type)
    start = start*1.0/100
    signal_knock = SignalTuk(mi, start=start - delta_time, end=start + delta_time, probe=probe)
    fig1 = px.line(signal_knock.signal,
                   height=200,
                  # title=f"{type} {day} {tree} {measurement} {probe} {start}"
                  )
    fig2 = px.line(signal_knock.fft,
                   height=300,
    # title=f"{type} {day} {tree} {measurement} {probe} {start}"
                  )
    for fig in [fig1,fig2]:
        fig.update_layout(
            xaxis_title=None,  # Skrývá popisek osy x
            yaxis_title=None,  # Skrývá popisek osy y
            showlegend=False  # Skrývá legendu
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0)  # l = left, r = right, t = top, b = bottom
        )

    fig2.update_yaxes(type="log")  # Logaritmická osa y
    return {'signal':fig1,'fft':fig2, 'text':f"{type}, {day}, {tree}, {measurement}, {probe}, {start}s",
            'index':index}


# Funkce pro stylování - přidání hranice, když se změní hodnota v úrovni 'tree'
def add_horizontal_line(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)

    # Projdi všechny řádky a přidej stylování
    for i in range(1, len(df)):
        if df.index[i][0] != df.index[i - 1][0]:  # Pokud se změní 'tree'
            styles.iloc[i, :] = 'border-top: 4px solid red'  # Přidej hranici

    return styles

@solara.component
@dynatree.timeit
def Tabulka():
    solara.Details(
        summary="Rozklikni pro popis",
        children = [solara.Markdown(
"""
* V tabulce jsou frekvence v Hz. Knock_time je čas v setinách sekundy od začátku měření. 
* Obarvení je po sloupcích. V každém sloupci jsou nejvyšší hodnoty nejtmavší.
* Pod tabulkou je median a další statistiky.
"""
        )]
    )
    subdf = (rdf.value
        .pipe(lambda d: d[d["tree"] == s.tree.value])
        .pipe(lambda d: d[d["day"] == s.day.value])
        .pipe(lambda d: d[d["type"] == s.method.value])
        #.pipe(lambda d: d[d["measurement"] == s.measurement.value])
        .drop(["day","tree","type","knock_index","filename"], axis=1)
        .pivot(index=['measurement', 'knock_time'], columns='probe', values='freq')
             )
    solara.display(
        subdf
        .style.format(precision=0).background_gradient(axis=0)
        .apply(add_horizontal_line, axis=None)
        .map(lambda x: 'color: lightgray' if pd.isnull(x) else '')
        .map(lambda x: 'background: transparent' if pd.isnull(x) else '')
    )
    with solara.Card(title="Statistics"):
        solara.display(subdf.describe())

first_portrait = solara.reactive(0)
def prev_ten():
    first_portrait.value = max(0,first_portrait.value - 10)
def next_ten(max):
    first_portrait.value = min(max,first_portrait.value + 10)
def prev_next_buttons(max):
    with solara.Row():
        solara.Button("Prev 10", on_click=prev_ten)
        solara.Button("Next 10", on_click=lambda: next_ten(max))

@solara.component
@dynatree.timeit
def Seznam():
    solara.Markdown(f"""
## {s.method.value} {s.day.value} {s.tree.value}    
    """)
    temp_df = (
        rdf.value
        .pipe(lambda d: d[d["tree"] == s.tree.value])
        .pipe(lambda d: d[d["day"] == s.day.value])
        .pipe(lambda d: d[d["type"] == s.method.value])
        #.drop(["day","tree","type","measurement","knock_index","filename"], axis=1)
    )
    pocet = len(temp_df)
    prev_next_buttons(max=pocet)
    if first_portrait.value > pocet - 2:
        first_portrait.value = pocet - 5
    for poradi, row in enumerate(temp_df.iterrows()):
        if poradi < first_portrait.value:
            continue
        if poradi > first_portrait.value+10:
            continue
        ReusableComponent(row, poradi, pocet)
    prev_next_buttons(max=pocet)

def change_OK_status(i):
    current = rdf.value.at[i,"valid"]
    rdf.value.at[i,"valid"] = not current
    rdf.value = rdf.value.copy()

@solara.component
def ReusableComponent(row, poradi, pocet):
    i, row = row
    is_valid = rdf.value.at[i, "valid"]
    if is_valid:
        style = {}
        bgstyle = {}
    else:
        style = {'border-right':'solid', 'border-color':'red', 'background-color':'lightgray'}
        bgstyle = {'background-color':'lightgray'}
    image_path = "/static/public/cache/" + row['filename'] + ".png"
    image_path_FFT = "/static/public/cache/FFT_" + row['filename'] + ".png"
    with solara.Card(style=style):
        with solara.Row(style=bgstyle):
            with solara.Column(style=bgstyle):
                solara.Text(f"{poradi + 1}/{pocet}, (id {i})")
                solara.Text(f"{row['measurement']} {row['probe']}")
                solara.Text(f"{row['knock_time']*1.0/100} sec")
                solara.Text(f"max at {round(row['freq'])} Hz")
                solara.Text(f"Manual peaks {rdf.value.at[i,"manual_peak"]} ")
            solara.Image(image_path)
            solara.Image(image_path_FFT)
        with solara.CardActions():
            s.measurement.value = row['measurement']
            solara.Button("Zobrazit grafy", text=True, color="primary", on_click=lambda:
                interactive_graph(s.method.value, s.day.value, s.tree.value, row['measurement'], row['probe'],
                              row['knock_time'],i)
                          )
            solara.Button("Změň status", text=True, color="primary", on_click=lambda:
                change_OK_status(i))
            if is_valid:
                solara.Text("✅")
            else:
                solara.Text("👎")

            # solara.Button("Action 2", text=True)

@solara.component
@dynatree.timeit
def Signal():
    solara.Markdown(
"""
## Signál akcelerometru

Akcelerometry a02_z a a02_x pro stanovení časů ťuknutí pro dvě sběrnice
""")

    summary_text = "Detailnější popis (klikni pro rozbalení)"
    additional_content = [solara.Markdown(
        """
        * Na obrázku je celý časový průbeh experimentu. Dva kanály a na nich pokusy o nalezení peaků pomocí find_peaks
        a nastavení thresholdů, prominencí a vzdáleností. 
    
        * Do tabulky se pro zpracování ukládají časy ťuků zaokrouhlené na setiny.  Takto je možné přidat později i 
          zatím nerozpoznané ťuky.
    
        * Je možné si zobrazit FFT transformaci, ale dlouho to trvá a lepší je pouužít předpočítané grafy na 
          vedlejší záložce.
        """)
    ]

    solara.Details(
        summary=summary_text,
        children=additional_content,
        expand=False
    )
    dynatree.logger.info("Signal entered")
    m = dynatree.DynatreeMeasurement(day=s.day.value, tree=s.tree.value, measurement=s.measurement.value,
                                     measurement_type=s.method.value)
    if m.data_acc5000 is None:
        solara.Error(f"Měření {m} není dostupné")
        return
    peak_timesA = find_peak_times_chanelA(m)
    peak_timesB = find_peak_times_chanelB(m)

    for peak_times ,probe in zip([peak_timesA, peak_timesB],["a02_z", "a02_x"]):
        df = m.data_acc5000.loc[:, probe]
        fig, ax = plt.subplots(figsize=(8,2))
        df.plot(ax=ax)
        ax.set(title=f"{m.day} {m.tree} {m.measurement} {m.measurement_type}, {probe}")
        for peak in peak_times:
            axvline(x=peak, color='red' ,linestyle='--', linewidth=1)
        ax.grid()
        plt.tight_layout()
        solara.FigureMatplotlib(fig, format='png')
    plt.close('all')
    solara.display(pd.DataFrame(peak_timesA, columns=["Čas ťuku, první sběrnice"]).T)
    solara.display(pd.DataFrame(peak_timesB, columns=["Čas ťuku, druhá sběrnice"]).T)
    plot_all(None, None)

@solara.component
def Rozklad():
    m = dynatree.DynatreeMeasurement(day=s.day.value, tree=s.tree.value, measurement=s.measurement.value,
                                     measurement_type=s.method.value)
    if m.data_acc5000 is None:
        solara.Error(f"Měření {m} není dostupné")
        return
    peak_timesA = find_peak_times_chanelA(m)
    peak_timesB = find_peak_times_chanelB(m)

    solara.Markdown(
    """
    ## Signál po ťuku

    FFT na intervalu 0.4 sekundy před a po ťuku.
    """)

    solara.ProgressLinear(plot_all.progress if plot_all.pending else False)

    if not plot_all.pending:
        solara.Info("Kliknutím na tlačítko se spustí generování obrázků. To může trvat dlouho.")

    # solara.Warning("Zatím se kreslí jenom jedna sběrnice, bez a02_x a bez a04. Je potřeba vyřešit automatické nalezení ťuků.")
    solara.Button("Plot or Replot", on_click=lambda : plot_all(m, [peak_timesA, peak_timesB]))

    # solara.ProgressLinear(plot_all.pending)
    # solara.display(peak_times.value)

    if plot_all.finished:
        images = plot_all.value
        for time, fig in images.items():
            # solara.Markdown(f"## Ťuk v čase {time}")
            with solara.Row():
                solara.FigureMatplotlib(fig[0], format='png')
                solara.FigureMatplotlib(fig[1], format='png')
        plt.close('all')
    else:
        if (plot_all.progress is not None) and (plot_all.progress>0):
            solara.SpinnerSolara(size="100px")
            if m.measurement == "M01":
                solara.Info("Generují se obrázky. Pro M01, výpočet trvá dlouho, i minutu.")
            else:
                solara.Info("Generují se obrázky. Ale mělo by to být rychlé.")


@task
@dynatree.timeit
def plot_all(m, peak_times):
    dynatree.logger.info(f"Plot All entered {m}")
    if peak_times is None:
        return {}
    answer = {}
    plot_all.progress = 0.001
    for number, p_times in enumerate(peak_times):
        if number == 0:
            probes = chanelA
            figsize = (3, 6)
        else:
            probes = chanelB
            figsize = (3, 4)
        dynatree.logger.info(f"probes is {probes}")
        n = len(p_times)
        for i,start in enumerate(p_times):
            plot_all.progress = (i) * 50.0/n +number*50
            dynatree.logger.warning(f"progress is {(i) * 50.0/n + number*50}")

            all_knocks = [SignalTuk(m, start=start-0.4, end=start+0.4, probe=probe) for probe in probes]
            signals = pd.concat([sg.signal for sg in all_knocks], axis=1)
            ax = signals.plot(subplots=True, sharex=True, figsize=figsize )
            [_.grid() for _ in ax]
            fig1 = plt.gcf()
            fig1.suptitle(f"{m.day} {m.tree} {m.measurement} {m.measurement_type}, ťuk v čase {start}s",
                          fontsize = 8)
            plt.tight_layout()

            ffts = pd.concat([i.fft for i in all_knocks], axis=1)
            dynatree.logger.info(f"ffts is {ffts.head()}")
            ax = ffts.plot(subplots=True, figsize=figsize, legend=False)
            ax = [_ for _ in ax]
            colnames = ffts.columns
            for i,axes in enumerate(ax):
                max_peak = ffts.iloc[5:,i].idxmax()
                dynatree.logger.info(f"max_peak is {max_peak}")
                axes.axvline(x=max_peak, color='red', linestyle='--')
                axes.text(
                    1.0, 0.0,  # Relativní souřadnice v pravém dolním rohu
                    f"{colnames[i]}: {round(max_peak)} Hz",  # Text popisku
                    ha = 'right',  # Zarovnání na pravý okraj textu
                    va = 'bottom',  # Přilepení k dolnímu okraji
                    fontsize = 8,  # Velikost textu
                    color = 'red',  # Barva textu
                    bbox = dict(facecolor='yellow'),  # Pozadí popisku
                    transform = axes.transAxes
                )
                axes.set(yscale='log')
                axes.grid()
            fig2 = plt.gcf()
            fig2.suptitle(f"{m.day} {m.tree} {m.measurement} {m.measurement_type}, ťuk v čase {start}s",
                        fontsize = 8)
            plt.tight_layout()

            answer[(start,number)] = [fig1, fig2]
    return answer

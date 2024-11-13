from matplotlib.pyplot import axvline
from dynatree import dynatree
import solara.lab
import solara
import pandas as pd
import dynatree.solara.select_source as s
import matplotlib.pyplot as plt
from solara.lab import task
from dynatree.signal_knock import SignalTuk, find_peak_times_chanelA, find_peak_times_chanelB, chanelA, chanelB
from dynatree_summary.acc_knocks import  delta_time
import solara.website
import logging
import plotly.express as px

from dynatree.solara.tahovky import interactive_graph

# dynatree.logger.setLevel(logging.INFO)
dynatree.logger.setLevel(logging.ERROR)

rdf =  solara.reactive(pd.read_csv("../outputs/FFT_acc_knock.csv"))

@solara.component
def Page():

    dynatree.logger.info("Page entered")
    solara.Title("DYNATREE: ACC ťuknutí")
    solara.Style(s.styles_css)
    with solara.Sidebar():
        s.Selection(
            optics_switch=False,
            report_optics_availability=False,
        )
    with solara.lab.Tabs():
        with solara.lab.Tab("Celé měření"):
            Signal()
            Rozklad()
        with solara.lab.Tab("Tabulka"):
            Tabulka()
        with solara.lab.Tab("Seznam"):
            with solara.Columns(1,1):
                with solara.Column():
                    Seznam()
                with solara.Column():
                    Graf()

@solara.component
def Graf():
    with solara.Card(style={'css':'sticky',  'top': '0'}):
        solara.ProgressLinear(interactive_graph.pending)
        if interactive_graph.finished:
            solara.FigurePlotly(interactive_graph.value[0])
            solara.FigurePlotly(interactive_graph.value[1])

@task
def interactive_graph(type, day, tree, measurement, probe, start):
    dynatree.logger.info("interactive graph entered")
    m = dynatree.DynatreeMeasurement(day=day, tree=tree, measurement=measurement,
                                     measurement_type=type)
    start = start*1.0/100
    signal_knock = SignalTuk(m, start=start - delta_time, end=start + delta_time, probe=probe)
    fig1 = px.line(signal_knock.signal,
                  # title=f"{type} {day} {tree} {measurement} {probe} {start}"
                  )
    fig2 = px.line(signal_knock.fft,
                  # title=f"{type} {day} {tree} {measurement} {probe} {start}"
                  )
    for fig in [fig1,fig2]:
        fig.update_layout(
            xaxis_title=None,  # Skrývá popisek osy x
            yaxis_title=None,  # Skrývá popisek osy y
            showlegend=False  # Skrývá legendu
        )
    fig2.update_yaxes(type="log")  # Logaritmická osa y
    return [fig1,fig2]

@solara.component
def Tabulka():
    solara.DataFrame(
        rdf.value
        .pipe(lambda d: d[d["tree"] == s.tree.value])
        .pipe(lambda d: d[d["day"] == s.day.value])
        .pipe(lambda d: d[d["type"] == s.method.value])
        .drop(["day","tree","type","measurement","knock_index","filename"], axis=1)
    )

first_portrait = solara.reactive(0)
def prev_ten():
    first_portrait.value = max(0,first_portrait.value - 10)
def next_ten():
    first_portrait.value = first_portrait.value + 10
def prev_next_buttons():
    with solara.Row():
        solara.Button("Prev 10", on_click=prev_ten)
        solara.Button("Next 10", on_click=next_ten)

@solara.component
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
    prev_next_buttons()
    for poradi, row in enumerate(temp_df.iterrows()):
        if poradi < first_portrait.value:
            continue
        if poradi > first_portrait.value+10:
            continue
        ReusableComponent(row, poradi, pocet)
    prev_next_buttons()

@solara.component
def ReusableComponent(row, poradi, pocet):
    i, row = row
    image_path = "/static/public/cache/" + row['filename'] + ".png"
    image_path_FFT = "/static/public/cache/FFT_" + row['filename'] + ".png"
    with solara.Card():
        with solara.Row():
            with solara.Column():
                solara.Text(f"{poradi + 1}/{pocet}")
                solara.Text(f"{row['measurement']} {row['probe']}")
                solara.Text(f"{row['knock_time']} * 0.01 sec")
                solara.Text(f"max at {round(row['freq'])} Hz")
            solara.Image(image_path)
            solara.Image(image_path_FFT)
        with solara.CardActions():
            solara.Button("Action 1", text=True, on_click=lambda:
            interactive_graph(s.method.value, s.day.value, s.tree.value, s.measurement.value, row['probe'],
                              row['knock_time'])
                          )
            solara.Button("Action 2", text=True)

@solara.component
def Signal():
    solara.Markdown(
"""
## Signál akcelerometru

Akcelerometry a02_z a a02_x pro stanovení časů ťuknutí pro dvě sběrnice
""")

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


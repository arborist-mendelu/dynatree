# from dynatree.find_measurements import available_measurements

from dynatree import dynatree
import solara.lab
import solara
import pandas as pd
import dynatree.solara.select_source as s
import matplotlib.pyplot as plt
from solara.lab import task
from dynatree.signal_knock import SignalTuk, find_peak_times

import logging
dynatree.logger.setLevel(logging.INFO)
# dynatree.logger.setLevel(logging.ERROR)


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
    Signal()
    Rozklad()

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
    for probe in ["a02_z", "a02_x"]:
        df = m.data_acc5000.loc[:, probe]
        fig, ax = plt.subplots(figsize=(8,3))
        df.plot(ax=ax)
        fig.suptitle(f"{m.day} {m.tree} {m.measurement} {m.measurement_type}, {probe}")
        ax.grid()
        plt.tight_layout()
        solara.FigureMatplotlib(fig, format='png')
    plt.close('all')
    plot_all(None, None, True)

@solara.component
def Rozklad():
    m = dynatree.DynatreeMeasurement(day=s.day.value, tree=s.tree.value, measurement=s.measurement.value,
                                     measurement_type=s.method.value)
    if m.data_acc5000 is None:
        solara.Error(f"Měření {m} není dostupné")
        return
    peak_timesA = find_peak_times(m)
    peak_timesB = find_peak_times(m, "a02_x", 6, shift=True)
    display(pd.DataFrame(peak_timesA, columns=["Čas ťuku, první sběrnice"]).T)
    display(pd.DataFrame(peak_timesB, columns=["Čas ťuku, druhá sběrnice"]).T)
    solara.Markdown(
    """
    ## Signál po ťuku

    Ťuknutí je stanoveno automaticky podle jedné z os jednoho akcelerometru. Po změně parametrů klikněte
    na tlačítko pro překreslení.
    """)

    solara.Warning("Zatím se kreslí jenom jedna sběrnice, bez a02_x a bez a04. Je potřeba vyřešit automatické nalezení ťuků.")
    solara.Button("Re-Plot", on_click=lambda : plot_all(m, [peak_timesA, peak_timesB]))
    solara.ProgressLinear(plot_all.progress if plot_all.pending else False)

    # solara.ProgressLinear(plot_all.pending)
    # solara.display(peak_times.value)


    if plot_all.not_called:
        plot_all(m, peak_times)
        return
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
                solara.Info("Generují se obrázky. Pro M01, výpočet trvá dlouho, něco přes 10 sekund.")
            else:
                solara.Info("Generují se obrázky. Ale mělo by to být rychlé.")


@task
@dynatree.timeit
def plot_all(m, peak_times, erase=False):
    dynatree.logger.info(f"Plot All entered {m}")
    answer = {}
    if erase:
        return answer

    peak_timesA = peak_times[0]
    peak_timesB = peak_times[1]
    plot_all.progress = 0.001
    for number, p_times in enumerate([peak_timesA, peak_timesB]):
        if number == 0:
            probes = ["a01_x", "a01_y", "a01_z", "a02_y", "a02_z", "a03_x", "a03_y", "a03_z", ]
        else:
            probes = ["a02_x", "a04_x", "a04_y", "a04_z"]
        dynatree.logger.info(f"probes is {probes}")
        n = len(p_times)
        for i,start in enumerate(p_times):
            # fig, ax = plt.subplots()
            # solara.Markdown(f"## Ťuk v čase {df.index[peak]:.2f}")
            # df.iloc[peak:peak + 100].plot(ax=ax)
            # solara.FigureMatplotlib(fig)
            plot_all.progress = (i) * 100.0/n
            dynatree.logger.info(f"progress is {(i + 1) * 100.0/n}")

            figsize = (7,12)
            all_knocks = [SignalTuk(m, start=start-0.4, end=start+0.4, probe=probe) for probe in probes]
            signals = pd.concat([sg.signal for sg in all_knocks], axis=1)
            # fig, ax = plt.subplots()
            ax = signals.plot(subplots=True, sharex=True, figsize=figsize )
            [_.grid() for _ in ax]
            fig1 = plt.gcf()
            fig1.suptitle(f"{m.day} {m.tree} {m.measurement} {m.measurement_type}, ťuk v čase {start}s")
            plt.tight_layout()

            ffts = pd.concat([i.fft for i in all_knocks], axis=1)
            dynatree.logger.info(f"ffts is {ffts.head()}")
            ax = ffts.plot(subplots=True, figsize=figsize)
            for i,axes in enumerate(ax):
                max_peak = ffts.iloc[5:,i].idxmax()
                dynatree.logger.info(f"max_peak is {max_peak}")
                axes.axvline(x=max_peak, color='red', linestyle='--')
                axes.set(yscale='log')
                axes.grid()
            fig2 = plt.gcf()
            fig2.suptitle(f"{m.day} {m.tree} {m.measurement} {m.measurement_type}, ťuk v čase {start}s")
            plt.tight_layout()

            answer[(start,number)] = [fig1, fig2]
    return answer


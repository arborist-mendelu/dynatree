# from dynatree.find_measurements import available_measurements
from dynatree import dynatree
import plotly.express as px
import solara.lab
import solara
import pandas as pd
import dynatree.solara.select_source as s
from plotly_resampler import FigureResampler
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from solara.lab import task
# import rich

# from dynatree.plot_fft_boxplots import plot_data
import logging
dynatree.logger.setLevel(logging.INFO)

class SignalTuk():

    def __init__(self, parent, start, end, probe):
        self.parent = parent
        self.start = start
        self.end = end
        self.signal = parent.data_acc5000.loc[start:end, probe]
        self.dt = 0.0002
        self.probe = probe

    @property
    def fft(self):
        N = self.signal.shape[0]  # get the number of points
        xf_r = fftfreq(N, self.dt)[:N // 2]
        yf = fft(self.signal.values)  # preform FFT analysis
        yf_r = 2.0 / N * np.abs(yf[0:N // 2])
        df_fft = pd.Series(index=xf_r, data=yf_r, name=self.probe)
        return df_fft

# def  hledani_peaku():


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

    solara.Markdown(
"""
## Signál akcelerometru

Akcelerometr a02_z pro stanovení časů ťuknutí
""")

    dynatree.logger.info("Hledani Peaku entered")
    m = dynatree.DynatreeMeasurement(day=s.day.value, tree=s.tree.value, measurement=s.measurement.value,
                                     measurement_type=s.method.value)
    if m.data_acc5000 is None:
        solara.Error(f"Měření {m} není dostupné")
        return
    df = m.data_acc5000.loc[:, "a02_z"]
    fig, ax = plt.subplots(figsize=(8,3))
    df.plot(ax=ax)
    fig.suptitle(f"{m.day} {m.tree} {m.measurement} {m.measurement_type}, a02_z")
    ax.grid()
    plt.tight_layout()
    solara.FigureMatplotlib(fig, format='png')

    peaks, _ = find_peaks(df.abs(), threshold=10, distance=75)
    peak_times = df.index[peaks]
    display(pd.DataFrame(peak_times, columns=["Čas ťuku"]).T)
    solara.Markdown(
    """
    ## Signál po ťuku

    Ťuknutí je stavnoveno automaticky podle jedné z os jednoho akcelerometru. Po změně parametrů klikněte
    na tlačítko pro překreslení.
    """)
    solara.Button("Plot", on_click=plot_all)
    solara.ProgressLinear(plot_all.pending)

    # solara.ProgressLinear(plot_all.pending)
    # solara.display(peak_times.value)


    if plot_all.not_called:
        plot_all()
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
        solara.SpinnerSolara(size="100px")
        if m.measurement == "M01":
            solara.Info("Generují se obrázky. Pro M01, výpočet trvá dlouho, několik sekund.")
        else:
            solara.Info("Generují se obrázky. Ale mělo by to být rychlé.")


@task
def plot_all():
    dynatree.logger.info("Plot All entered")
    m = dynatree.DynatreeMeasurement(day=s.day.value, tree=s.tree.value, measurement=s.measurement.value,
                                     measurement_type=s.method.value)
    df = m.data_acc5000.loc[:, "a02_z"]
    peaks, _ = find_peaks(df.abs(), threshold=10, distance=75)
    peak_times = df.index[peaks]

    answer = {}
    for start in peak_times:
        # fig, ax = plt.subplots()
        # solara.Markdown(f"## Ťuk v čase {df.index[peak]:.2f}")
        # df.iloc[peak:peak + 100].plot(ax=ax)
        # solara.FigureMatplotlib(fig)

        probes = ["a02_y", "a02_z", "a03_y", "a03_z"]
        a = [SignalTuk(m, start=start, end=start+0.1, probe=probe) for probe in probes]
        signals = pd.concat([sg.signal for sg in a], axis=1)
        # fig, ax = plt.subplots()
        ax = signals.plot(subplots=True, sharex=True )
        [_.grid() for _ in ax]
        fig1 = plt.gcf()
        fig1.suptitle(f"{m.day} {m.tree} {m.measurement} {m.measurement_type}, at {start}s")
        plt.tight_layout()

        ffts = pd.concat([i.fft for i in a], axis=1)
        ax = ffts.plot(subplots=True)
        for a in ax:
            a.set(yscale='log')
            a.grid()
        fig2 = plt.gcf()
        fig2.suptitle(f"{m.day} {m.tree} {m.measurement} {m.measurement_type}, at {start}s")
        plt.tight_layout()

        answer[start] = [fig1, fig2]
    return answer


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
dynatree.logger.setLevel(logging.ERROR)

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

peak_times = []

@solara.component
def  hledani_peaku(m,df):
    global peak_times
    dynatree.logger.info("Hledani Peaku entered")
    fig, ax = plt.subplots()
    df.plot(ax=ax)
    ax.grid()
    solara.FigureMatplotlib(fig)

    peaks, _ = find_peaks(df.abs(), threshold=10, distance=75)
    peak_times = df.index[peaks]
    display(pd.DataFrame(peak_times, columns=["Čas ťuku"]).T)
    solara.Markdown(
    """
    ## Signál po ťuku

    Ťuknutí je stavnoveno automaticky podle jedné z os jednoho akcelerometru.
    """)

    # solara.ProgressLinear(plot_all.pending)
    # solara.display(peak_times.value)

    answer = plot_all(m, df)
    for time, fig in answer.items():
        # solara.Markdown(f"## Ťuk v čase {time}")
        with solara.Row():
            solara.FigureMatplotlib(fig[0])
            solara.FigureMatplotlib(fig[1])
    plt.close('all')

@solara.component
def Page():

    dynatree.logger.info("Page entered")
    solara.Title("DYNATREE: ACC ťuknutí")
    solara.Style(s.styles_css)
    with solara.Sidebar():
        s.Selection()

    solara.Markdown(
"""
## Signál akcelerometru

Akcelerometr a02_z pro stanovení časů ťuknutí
""")


    m = dynatree.DynatreeMeasurement(day=s.day.value, tree=s.tree.value, measurement=s.measurement.value,
                                     measurement_type=s.method.value)
    df = m.data_acc5000.loc[:, "a02_z"]

    hledani_peaku(m, df)


def plot_all(m, df):
    dynatree.logger.info("Plot All entered")
    answer = {}
    print(peak_times)
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

        ffts = pd.concat([i.fft for i in a], axis=1)
        ax = ffts.plot(subplots=True)
        for a in ax:
            a.set(yscale='log')
            a.grid()
        fig2 = plt.gcf()
        fig2.suptitle(f"{m.day} {m.tree} {m.measurement} {m.measurement_type}, at {start}s")
        answer[start] = [fig1, fig2]
    return answer


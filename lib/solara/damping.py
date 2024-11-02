import solara

import lib.solara.select_source as s
from lib_dynatree import DynatreeMeasurement
from lib_damping import DynatreeDampedSignal
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import lib_dynatree
import logging
lib_dynatree.logger.setLevel(logging.ERROR)

def draw_signal_with_envelope(s, fig, envelope=None, k=0, q=0, row=1, col=1 ):
    signal, time = s.damped_signal.reshape(-1), s.damped_time
    x = time
    y = np.exp(k * time + q)
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                             y=np.concatenate([y, -y[::-1]]),
                             fill='toself',
                             fillcolor='lightblue',
                             line=dict(color='lightblue'),
                             showlegend=False), row=row, col=col)
    fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name='signal', line=dict(color='blue')), row=row, col=col)
    if envelope is not None:
        fig.add_trace(
            go.Scatter(x=time, y=envelope, mode='lines', name='envelope', line=dict(color='red'), legendgroup='obalka'), row=row, col=col)
        fig.add_trace(go.Scatter(x=time, y=-envelope, mode='lines', showlegend=False, line=dict(color='red'),
                                 legendgroup='obalka'), row=row, col=col)
    # fig.update_layout(xaxis_title="Čas", yaxis_title="Signál")
    return fig

def resetuj(x=None):
    # Srovnani(resetuj=True)
    s.measurement.set(s.measurements.value[0])
    # generuj_obrazky()

data_source = solara.reactive("Pt3")
data_sources = ["Pt3", "Pt4", "a01_z", "a02_z", "a03_z", "a01_y", "a02_y", "a03_y"]

@solara.component()
def Page():
    solara.Title("DYNATREE: Damping")
    solara.Style(s.styles_css)
    with solara.Sidebar():
        with solara.Sidebar():
            s.Selection(exclude_M01=True,
                        optics_switch=False,
                        day_action=resetuj,
                        tree_action=resetuj,
                        # measurement_action=generuj_obrazky
                        )
            # s.ImageSizes()
            with solara.Card(title="Signal source choice"):
                solara.ToggleButtonsSingle(value=data_source, values=data_sources)
    # draw_images()
    try:
        draw_images()
    except:
        solara.Warning("Někde nastala chyba.")

def draw_images():

    m = DynatreeMeasurement(day=s.day.value,
                            tree=s.tree.value,
                            measurement=s.measurement.value,
                            measurement_type=s.method.value)
    if "Pt" in data_source.value:
        dt = 0.01
    else:
        dt = 0.0002
    sig = DynatreeDampedSignal(m, data_source.value, dt=dt)

    data = {}
    fig = make_subplots(rows=3, cols=1, shared_xaxes='all', shared_yaxes='all')
    envelope, k, q = sig.hilbert_envelope.values()
    fig = draw_signal_with_envelope(sig, fig, envelope, k, q, row=1)
    data['hilbert'] = [k]

    peaks, k, q = sig.fit_maxima.values()
    fig = draw_signal_with_envelope(sig, fig, k=k, q=q, row=2)
    fig.add_trace(go.Scatter(x=peaks.index, y=peaks.values.reshape(-1),
                             mode='markers', name='peaks', line=dict(color='red')), row=2, col=1)
    data['extrema'] = [k]

    envelope, k, q, freq, fft_data = sig.wavelet_envelope.values()
    fig = draw_signal_with_envelope(sig, fig, envelope, k=k, q=q, row=3)
    data['wavelets'] = [k]

    fig.update_layout(title=f"Proložení exponenciely pomocí několika metod",
                      height = 800,
                      )

    fig.update_yaxes(title_text="Hilbert", row=1, col=1)
    fig.update_yaxes(title_text="Maxima/minima", row=2, col=1)
    fig.update_yaxes(title_text="Wavelet", row=3, col=1)

    df = pd.DataFrame.from_dict(data)
    df.index = ['k']
    solara.display(df)
    solara.FigurePlotly(fig)


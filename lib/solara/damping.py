import solara

import lib.solara.select_source as s
from lib_dynatree import DynatreeMeasurement
from lib_damping import DynatreeDampedSignal
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

import lib_dynatree
import logging
lib_dynatree.logger.setLevel(logging.INFO)

def draw_signal_with_envelope(s, envelope=None, k=0, q=0, ):
    signal, time = s.damped_signal.reshape(-1), s.damped_time
    fig = go.Figure()
    x = time
    y = np.exp(k * time + q)
    fig.add_trace(go.Scatter(x=np.concatenate([x, x[::-1]]),
                             y=np.concatenate([y, -y[::-1]]),
                             fill='toself',
                             fillcolor='lightblue',
                             line=dict(color='lightblue'),
                             showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name='signal', line=dict(color='blue')))
    if envelope is not None:
        fig.add_trace(
            go.Scatter(x=time, y=envelope, mode='lines', name='envelope', line=dict(color='red'), legendgroup='obalka'))
        fig.add_trace(go.Scatter(x=time, y=-envelope, mode='lines', showlegend=False, line=dict(color='red'),
                                 legendgroup='obalka'))
    fig.update_layout(xaxis_title="Čas", yaxis_title="Signál")
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
    draw_images()
    # try:
    #     draw_images()
    # except:
    #     solara.Warning("Zatím jenom optika a Pt3. Ostatní měření a ostatní proby budou později.")

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

    with solara.ColumnsResponsive(default=12, large=6, xlarge=4, wrap=True):
        with solara.Card():
            pass
            envelope, k, q = sig.hilbert_envelope.values()
            fig = draw_signal_with_envelope(sig, envelope, k, q)
            fig.update_layout(title=f"Hilbertova obálka, k={k:.4f}<br>{m}",
                              # height=s.height.value,
                              # width=s.width.value,
                              )
            solara.FigurePlotly(fig)

        with solara.Card():
            pass
            peaks, k, q = sig.fit_maxima.values()
            fig = draw_signal_with_envelope(sig, k=k, q=q)
            fig.add_trace(go.Scatter(x=peaks.index, y=peaks.values.reshape(-1),
                                     mode='markers', name='peaks', line=dict(color='red')))
            fig.update_layout(title=f"Proložení exponenciely podle peaků, k={k:.4f}<br>{m}",
                # height = s.height.value,
                # width = s.width.value,
                )
            solara.FigurePlotly(fig)

        with solara.Card():
            # if "Pt" in data_source.value:
            envelope, k, q, freq, fft_data = sig.wavelet_envelope.values()
            fig = draw_signal_with_envelope(sig,k=k, q=q)
            fig.update_layout(title=f"Proložení exponenciely pomocí waveletu, k={k:.4f}<br>{m}",
                # height = s.height.value,
                # width = s.width.value,
                )
            solara.FigurePlotly(fig)

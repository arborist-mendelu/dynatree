import solara
from solara.lab import task

import dynatree.solara.select_source as s
from dynatree.dynatree import DynatreeMeasurement
from dynatree.damping import DynatreeDampedSignal
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from dynatree import dynatree
import logging
import time
import config

dynatree.logger.setLevel(dynatree.logger_level)
dynatree.logger.setLevel(logging.ERROR)

df_failed = pd.read_csv(config.file["FFT_failed"]).values.tolist()

loading_start = time.time()
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
        env_time = time
        if isinstance(envelope, pd.Series):
            dynatree.logger.info("Series to data in envelope function")
            env_time = envelope.index
            envelope = envelope.values
        fig.add_trace(
            go.Scatter(x=env_time, y=envelope, mode='lines', name='envelope', line=dict(color='red'), legendgroup='obalka'), row=row, col=col)
        fig.add_trace(go.Scatter(x=env_time, y=-envelope, mode='lines', showlegend=False, line=dict(color='red'),
                                 legendgroup='obalka'), row=row, col=col)
    # fig.update_layout(xaxis_title="Čas", yaxis_title="Signál")
    return fig

def resetuj(x=None):
    s.measurement.set(s.measurements.value[0])
    draw_images()

data_source = solara.reactive("Elasto(90)")
data_sources = ["Elasto(90)", "blueMaj", "yellowMaj", "Pt3", "Pt4", "a01_z", "a02_z", "a03_z", "a01_y", "a02_y", "a03_y"]

@solara.component()
def Page():
    solara.Title("DYNATREE: Damping")
    solara.Style(s.styles_css)
    with solara.Sidebar():
        s.Selection(exclude_M01=True,
                    optics_switch=False,
                    day_action=resetuj,
                    tree_action=resetuj,
                    measurement_action=draw_images
                    )
        # s.ImageSizes()
    damping_graphs()

@solara.component()
def damping_graphs():
    with solara.Sidebar():
        with solara.Card(title="Signal source choice"):
            solara.ToggleButtonsSingle(value=data_source, values=data_sources, on_value = draw_images)
    solara.ProgressLinear(draw_images.pending)
    coords = [s.tree.value, s.day.value, s.method.value, s.measurement.value, data_source.value]
    solara.Markdown(f"## {" ".join(coords)}")
    if draw_images.not_called:
        draw_images()
        return
    if not draw_images.finished:
        solara.Error("""
        Probíhá výpočet. Trvá řádově vteřiny.Pokud se tato zpráva zobrazuje déle, něco je špatně. 
        Možná je vybrán probe pro optiku, ale optika není k dispozici. 
        V takovém případě vyber jiný den nebo jiný probe.
        """)
        solara.SpinnerSolara(size="100px")
        return
    ans = draw_images.value
    if ans is None:
        solara.Error("Nekde nastala chyba")
    else:
        df, fig, marked_failed = ans
        if marked_failed == True:
            solara.Error(f"This measurement was marked as failed.")
        solara.display(df)
        solara.FigurePlotly(fig)

@task
def draw_images(temp=None):
    m = DynatreeMeasurement(day=s.day.value,
                            tree=s.tree.value,
                            measurement=s.measurement.value,
                            measurement_type=s.method.value)
    dynatree.logger.info(f"Measurement {m}")
    signal_source = data_source.value
    if "Pt" in data_source.value:
        dt = 0.01
        if not m.is_optics_available:
            solara.Warning(f"No optics for {m}")
            print(f"Neni optika pro {m}")
            return None
    elif "A" in data_source.value:
        dt = 0.0002
    else:
        dt = 0.12
        if data_source.value in ["blueMaj", "yellowMaj"]:
            pass
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
    maximum = np.argmax(envelope)
    # dynatree.logger.info(f"Maximum obalky je pro {maximum}")
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
    return df, fig, sig.marked_failed

dynatree.logger.info(f"File damping.py loaded in {time.time()-loading_start} sec.")
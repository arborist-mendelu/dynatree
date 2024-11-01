import solara

import lib.solara.select_source as s
from lib_dynatree import DynatreeMeasurement
from lib.damping import DynatreeDampedSignal, draw_signal_with_envelope
import plotly.graph_objects as go

import lib_dynatree
import logging
lib_dynatree.logger.setLevel(logging.INFO)

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
    # try:
    draw_images()
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

    with solara.ColumnsResponsive(default=6, large=4, wrap=True):
        with solara.Card():
            envelope, k, q = sig.hilbert_envelope.values()
            fig = draw_signal_with_envelope(sig, envelope, k, q)
            fig.update_layout(title=f"Hilbertova obálka, k={k:.4f}<br>{m}",
                              # height=s.height.value,
                              # width=s.width.value,
                              )
            solara.FigurePlotly(fig)

        with solara.Card():
            peaks, k, q = sig.fit_maxima.values()
            fig = draw_signal_with_envelope(sig, k=k, q=q)
            fig.add_trace(go.Scatter(x=peaks.index, y=peaks.values, mode='markers', name='peaks', line=dict(color='red')))
            fig.update_layout(title=f"Proložení exponenciely podle peaků, k={k:.4f}<br>{m}",
                # height = s.height.value,
                # width = s.width.value,
                )
            solara.FigurePlotly(fig)

        with solara.Card():
            if "Pt" in data_source.value:
                envelope, k, q, freq, fft_data = sig.wavelet_envelope.values()
                fig = draw_signal_with_envelope(sig,k=k, q=q)
                fig.update_layout(title=f"Proložení exponenciely pomocí waveletu, k={k:.4f}<br>{m}",
                    # height = s.height.value,
                    # width = s.width.value,
                    )
                solara.FigurePlotly(fig)

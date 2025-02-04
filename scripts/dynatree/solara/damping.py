import solara
from solara.lab import task
import dynatree.solara.select_source as s
from dynatree.dynatree import DynatreeMeasurement
from dynatree.dynatree import timeit
from dynatree.damping import DynatreeDampedSignal
from dynatree.peak_width import find_peak_width
from dynatree.FFT import df_failed_FFT_experiments, DynatreeSignal
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import pandas as pd
from dynatree import dynatree
import logging
import time
import config
import matplotlib.pyplot as plt
from solara.lab.components.confirmation_dialog import ConfirmationDialog
from dynatree.solara.snackbar import show_snack, snack

dynatree.logger.setLevel(dynatree.logger_level)
dynatree.logger.setLevel(logging.ERROR)

df_failed = pd.read_csv(config.file["FFT_failed"]).values.tolist()
loading_start = time.time()


def draw_signal_with_envelope(s, fig, envelope=None, k=0, q=0, row=1, col=1):
    signal, time = s.damped_signal.reshape(-1), s.damped_time
    x = time
    if k is not None:
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
        if isinstance(envelope, list):
            env_time = envelope[0]
            envelope = envelope[1]
        fig.add_trace(
            go.Scatter(x=env_time, y=envelope, mode='lines', name='envelope', line=dict(color='red'),
                       legendgroup='obalka'), row=row, col=col)
        fig.add_trace(go.Scatter(x=env_time, y=-envelope, mode='lines', showlegend=False, line=dict(color='red'),
                                 legendgroup='obalka'), row=row, col=col)
    # fig.update_layout(xaxis_title="Čas", yaxis_title="Signál")
    return fig


def resetuj(x=None):
    s.measurement.set(s.measurements.value[0])
    draw_images()

def resetuj2(x=None):
    s.measurement.set(s.measurements.value[0])
    do_find_peaks()

data_source = solara.reactive("Elasto(90)")
devices = {'pulling': ["Elasto(90)", "blueMaj", "yellowMaj"],
           'optics': ["Pt3", "Pt4"],
           'acc': ["a01_z", "a02_z", "a03_z", "a01_y", "a02_y",  "a03_y"]}
data_sources = sum(devices.values(), [])


@solara.component
def Page():
    solara.Title("DYNATREE: Damping")
    solara.Style(s.styles_css)
    snack()
    with solara.lab.Tabs(lazy=True):
        with solara.lab.Tab("From amplitudes"):
            with solara.Sidebar():
                s.Selection(exclude_M01=True,
                            optics_switch=False,
                            day_action=resetuj,
                            tree_action=resetuj,
                            measurement_action=draw_images
                            )
                # s.ImageSizes()
            try:
                damping_graphs()
            except:
                solara.Error("Some problem appeared")
        with solara.lab.Tab("From FFT (images)"):
            with solara.Sidebar():
                s.Selection(exclude_M01=True,
                            optics_switch=False,
                            day_action=resetuj2,
                            tree_action=resetuj2,
                            measurement_action= do_find_peaks
                            )
                # s.ImageSizes()
            try:
                peak_width_graph()
            except:
                pass
        with solara.lab.Tab("From FFT (tables)"):
            with solara.Sidebar():
                with solara.Card(title="Background gradient"):
                    solara.ToggleButtonsSingle(value=gradient_axis, values=gradient_axes)
            # try:
            peak_width_table()
        # except:
        #     solara.Error("Some problem appeared")

current = {'from_amplitudes': None, 'from_fft': None}
gradient_axis = solara.reactive("Columns")
gradient_axes = ["Rows", "Columns", "Table"]
@solara.component
def peak_width_table():
    df = pd.read_csv(config.file['outputs/peak_width'])
    df = df.pivot(index=df.columns[:4], columns="probe", values="width").drop(["a04_y", "a04_z"], axis=1)
    trees = df.index.get_level_values('tree').drop_duplicates()


    if gradient_axis.value == "Rows":
        axis = 1
    elif gradient_axis.value == "Columns":
        axis = 0
    else:
        axis = None
    for tree in trees:
        _ = (
            df[df.index.get_level_values('tree') == tree]
            .style.format(precision=3).background_gradient(axis=axis)
            .map(lambda x: 'color: lightgray' if pd.isnull(x) else '')
            .map(lambda x: 'background: transparent' if pd.isnull(x) else '')
        )
        with solara.Card(title=f"Tree {tree}"):
            solara.display(_)

@solara.component
def peak_width_graph():
    global current
    with solara.Sidebar():
        with solara.Column():
            solara.Button("Plot/Replot", on_click=do_find_peaks, color='primary')
    coords = [s.tree.value, s.day.value, s.method.value, s.measurement.value]
    if current['from_fft'] != coords:
        current['from_fft'] = coords
        do_find_peaks()
    solara.Markdown(f"## {" ".join(coords)}")
    solara.Info(
        f"Relative peak width (peak width at given height divided by the peak position). Click Plot/Replot for another measurement. It takes few seconds to draw all sensors.")
    solara.ProgressLinear(find_peak_widths.pending)

    if not find_peak_widths.finished:
        return
    with solara.Row(style={'flex-wrap': 'wrap'}):
        for target_probe, ans in zip(data_sources, find_peak_widths.value):
            coordsf = [s.method.value, s.day.value, s.tree.value, s.measurement.value, target_probe]
            if ans is None:
                continue
            with solara.Card(title=f"{target_probe}: {round(ans['width'], 4)}", style={'min-width': '150px'}):
                if coordsf in df_failed_FFT_experiments.values.tolist():
                    solara.Error("This measurement has been marked as failed.")
                try:
                    solara.FigureMatplotlib(ans['fig'])
                except:
                    pass

                def create_button(label, target_probe, output):
                    # Funkce vytvoří tlačítko a uzavře aktuální hodnotu target_probe
                    return solara.Button(label=label, on_click=lambda x=None: open_dialog(target_probe, output=output))

                # Vytvoření tlačítek
                create_button("Show experiment", target_probe, output='experiment')
                create_button("Show signal for FFT", target_probe, output='signal')
                create_button("Show FFT spectrum", target_probe, output='fft')

                # solara.Button(label="Show signal", on_click=lambda x=None: open_dialog(target_probe, output='signal'))
                # solara.Button(label="Show FFT spectrum",
                #               on_click=lambda x=None: open_dialog(target_probe, output='fft'))
    plt.close('all')
    create_overlay()


@solara.component
def create_overlay():
    with ConfirmationDialog(show_dialog.value, on_ok=close_dialog, on_cancel=close_dialog, max_width='90%',
                            title=""):
        if open_dialog.finished:
            ans = open_dialog.value
            solara.Markdown(f"## {" ".join(ans['coords'])}")
            ans['data'].name = "value"
            fig = ans['data'].plot(backend='plotly')
            if ans['output'] == 'fft':
                fig.update_layout(
                    yaxis=dict(type="log"),
                    xaxis=dict(range=[0, 10]),
                    height=500,
                    width=700,
                )
            solara.FigurePlotly(fig)


show_dialog = solara.reactive(False)

def close_dialog():
    show_dialog.value = False

@task
def open_dialog(probe, output):
    m = DynatreeMeasurement(day=s.day.value,
                            tree=s.tree.value,
                            measurement=s.measurement.value,
                            measurement_type=s.method.value)
    show_dialog.value = True
    coords = [s.day.value, s.method.value, s.tree.value, s.measurement.value, probe]
    if output=='experiment':
        ans = m.signal(senzor=probe)
        return {'coords':coords, 'data':ans, 'output':output}
    sig = DynatreeSignal(m, signal_source=probe, tukey=0.1)
    if output=='signal':
        return {'coords':coords, 'data':sig.signal, 'output':output}
    return {'coords':coords, 'data':sig.fft, 'output':output}

def do_find_peaks(x=None):
    show_snack(text = "This computation may take some time!")
    m = DynatreeMeasurement(day=s.day.value,
                            tree=s.tree.value,
                            measurement=s.measurement.value,
                            measurement_type=s.method.value)
    find_peak_widths(m)


@task
@timeit
def find_peak_widths(m):
    return [find_peak_width(m, sensor=target_probe, save_fig=True) for target_probe in data_sources]

def create_overlay_signal(output):
    show_dialog.value = True
    return None

@solara.component
def damping_graphs():
    # global current
    with solara.Sidebar():
        with solara.Card(title="Signal source choice"):
            solara.ToggleButtonsSingle(value=data_source, values=data_sources, on_value=draw_images)
        with solara.Card(title="Plot data"):
            solara.Button("Signal", on_click=lambda :create_overlay_signal('signal'))
            # solara.Button("FFT", on_click=lambda :create_overlay_signal('fft'))

    with ConfirmationDialog(show_dialog.value, on_ok=close_dialog, on_cancel=close_dialog, max_width='90%',
                            title=""):
        if show_dialog.value:
            solara.Markdown(f"## {s.day.value} {s.tree.value} {s.measurement.value} {s.method.value} {data_source.value}")
            m = DynatreeMeasurement(day=s.day.value, tree=s.tree.value, measurement=s.measurement.value, measurement_type=s.method.value)
            fig, ax = plt.subplots()
            if data_source.value in devices['pulling']:
                curr_data = m.data_pulling[data_source.value]
            elif data_source.value in devices['optics']:
                curr_data = m.data_optics[(data_source.value,"Y0")]
            else:
                curr_data = m.data_acc5000[data_source.value]
            curr_data.plot(ax=ax)
            solara.FigureMatplotlib(fig)
    solara.ProgressLinear(draw_images.pending)
    coords = [s.tree.value, s.day.value, s.method.value, s.measurement.value, data_source.value]
    solara.Markdown(f"## {" ".join(coords)}")
    # if current['from_amplitudes'] != coords:
    #     current['from_amplitudes'] = coords
    #     dynatree.logger.info("Graphs from peaks are not current. Calling draw_images.")
    #     draw_images()
    #     # return
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
        df = ans['df']
        fig = ans['fig']
        marked_failed = ans['failed']
        background_color = 'transparent'
        if marked_failed == True:
            solara.Error(f"This measurement was marked as failed.")
            background_color = '#f8d7da'
        with solara.Card(style={'background-color': background_color}):
            T = 1/ans['peak']
            with solara.Row():
                df.loc["LDD",:] = df.loc['b',:] * T

                # Vlastní funkce pro formátování
                def custom_format(x):
                    if abs(x) < 1e-5 and x != 0:  # Pokud je číslo menší než 1e-5 (kromě nuly), zobrazí se ve vědeckém formátu
                        return f"{x:.2e}"
                    else:  # Jinak zobrazí 8 desetinných míst
                        return f"{x:.8f}"
                solara.display(df.style.format(custom_format))
                with solara.Column():
                    solara.Markdown("The signal envelope is $e^{-bt}$.")
                    solara.Text(f"Main freq is f={ans['peak']:.5} Hz, period is T={T:.5} s.")
                    solara.Text("LDD = b*T")
                    solara.Text("The length of the anlayzed time interval is ... TODO")

            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',  # Pozadí celého plátna
                              #plot_bgcolor='rgba(0,0,0,0)'
                              )
            solara.FigurePlotly(fig)
        with solara.Warning():
            solara.Markdown("""
            TODO: 

            * Nepoužívat příliš dlouhý časový interval. Konec nastavit na 25% maxima. Zatím je nastaveno 
              u metody využívající extrémy. Hilbert a wavelety tuto informaci přebírají. Je to tak dostatečné?
            * Možná bude potřeba opravit hledání peaků a další parametry pro optiku a akcelerometry.   
            * Možná bude potřeba doladit vycentrování signálu tak, aby hilbert měl co nejmenší zvlnění.     
            """, style={'color': 'inherit'})
        with solara.Info():
            solara.Markdown(
                """
                * **Extrémy**: 
                    * Po vypuštění se vynechá perioda. 
                    * Peaky nesmí být blíže než 75 procent periody.
                    * Po prvním peaku, který je pod 25 procent maxima se signál už neuvažuje.
                * **Hilbert**
                    * Analyzovaný časový úsek stejný jako u metody extrémů.
                    * Zvážit, jestli by se nedalo zvlnění ovlivnit odstraněním trendu.
                * **Wavelet**
                    * Analyzovaný časový úsek stejný jako u metody extrémů.
                * V tabulce jsou i další metriky z lineární regrese, ale hilbertova obálka i wavelet používají 
                  řádově jiný počet bodů a proto není možné srovnávat například p-hodnoty.
                """, style={'color':'inherit'}
            )


@task
@timeit
def draw_images(temp=None):

    m = DynatreeMeasurement(day=s.day.value,
                            tree=s.tree.value,
                            measurement=s.measurement.value,
                            measurement_type=s.method.value)
    dynatree.logger.info(f"Measurement {m}")
    signal_source = data_source.value
    if "Pt" in data_source.value:
        dynatree.logger.info(f"Damping, optics {data_source.value}")
        dt = 0.01
        if not m.is_optics_available:
            solara.Warning(f"No optics for {m}")
            return None
    elif "a" in data_source.value[:2]:
        dynatree.logger.info(f"Damping, accelerometer {data_source.value}")
        dt = 0.0002
    else:
        dynatree.logger.info(f"Damping, pulling {data_source.value}")
        dt = 0.12
    sig = DynatreeDampedSignal(m, data_source.value, dt=dt)

    data = {}
    fig = make_subplots(rows=3, cols=1, shared_xaxes='all', shared_yaxes='all')
    envelope, k, q, R2, p_value, std_err = sig.hilbert_envelope.values()
    fig = draw_signal_with_envelope(sig, fig, envelope, k, q, row=1)
    data['hilbert'] = [None if k is None else -k, R2, p_value, std_err]

    peaks, k, q, R2, p_value, std_err = sig.fit_maxima().values()
    fig = draw_signal_with_envelope(sig, fig, k=k, q=q, row=2)
    fig.add_trace(go.Scatter(x=peaks.index, y=peaks.values.reshape(-1),
                             mode='markers', name='peaks', line=dict(color='red')), row=2, col=1)
    data['extrema'] = [None if k is None else -k, R2, p_value, std_err ]

    envelope, k, q, freq, fft_data, R2, p_value, std_err = sig.wavelet_envelope.values()
    fig = draw_signal_with_envelope(sig, fig, envelope, k=k, q=q, row=3)
    data['wavelets'] = [None if k is None else -k, R2, p_value, std_err ]

    fig.update_layout(title=f"Proložení exponenciely pomocí několika metod",
                      height=800,
                      )

    fig.update_yaxes(title_text="Hilbert", row=1, col=1)
    fig.update_yaxes(title_text="Maxima/minima", row=2, col=1)
    fig.update_yaxes(title_text="Wavelet", row=3, col=1)

    df = pd.DataFrame.from_dict(data)
    df.index = ['b','R^2','p_value','std_err']

    return {'df':df, 'fig':fig, 'failed':sig.marked_failed, 'peak':sig.main_peak}


dynatree.logger.info(f"File damping.py loaded in {time.time() - loading_start} sec.")

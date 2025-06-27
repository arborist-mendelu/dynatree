import solara
from solara.lab import task
import dynatree.solara.select_source as s
from dynatree.dynatree import DynatreeMeasurement
from dynatree.dynatree import timeit, get_zero_rating
from dynatree.damping import DynatreeDampedSignal
from dynatree.peak_width import find_peak_width
from dynatree.FFT import df_failed_FFT_experiments, DynatreeSignal
import plotly.graph_objects as go
import plotly.express as px
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
from dynatree.dynatree_util import add_horizontal_line
import requests
import ipywidgets as widgets
from dynatree.solara.code import show_load_code

dynatree.logger.setLevel(dynatree.logger_level)
dynatree.logger.setLevel(logging.ERROR)

df_failed = pd.read_csv(config.file["FFT_failed"]).values.tolist()
loading_start = time.time()

def draw_signal_with_envelope(s, fig, envelope=None, k=0, q=0, row=1, col=1):
    signal, time = s.damped_signal.reshape(-1), s.damped_time
    x = time
    if k is not None:
        y = np.exp(-k * time + q)
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

def nuluj_a_resetuj(x=None):
    manual_signal_end.value = None
    resetuj()


# def resetuj2(x=None):
#     s.measurement.set(s.measurements.value[0])
#     do_find_peaks()

data_source = solara.reactive("Elasto(90)")
devices = {'pulling': ["Elasto(90)", "blueMaj", "yellowMaj"],
           'optics': ["Pt3", "Pt4"],
           'acc': ["a01_z", "a02_z", "a03_z", "a01_y", "a02_y",  "a03_y"]}
data_sources = sum(devices.values(), [])
damping_parameter = solara.reactive("LDD")
damping_parameters = ["b","LDD"]

filtr_R_min = solara.reactive(-1)
filtr_R_max = solara.reactive(1)
filtr_T_min = solara.reactive(0)
filtr_T_max = solara.reactive(1000)
switch_damping = solara.reactive("defmulti")
data_selection = solara.reactive("optimistic")
data_selection_types = ["all", "optimistic", "pesimistic"]
manual_signal_end = solara.reactive(None)

tab_index = solara.reactive(1)

@solara.component
def Page():
    solara.Title("DYNATREE: Damping")
    styles_css = s.styles_css + """
        .image-preview {
            color: #007bff;
            text-decoration: none;
            cursor: pointer;
            position: relative;
        }

        .preview-container {
            position: fixed;
            top: 10px;
            right: 10px;
            display: none;
            z-index: 1000;
            background: white;
            border: 1px solid #ccc;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.2);
            padding: 5px;
        }

        .preview-container img {
            max-width: 800px;
            height: auto;
            display: block;
            
        #startBtn {
            background-color: lightgray !important;
            padding: 5px;
        }       
        
    """
    solara.Style(styles_css)
    snack()
    with solara.lab.Tabs(lazy=True, value=tab_index):
        with solara.lab.Tab("From amplitudes (single measurement)"):
            with solara.Sidebar():
                s.Selection(exclude_M01=True,
                            optics_switch=False,
                            day_action=nuluj_a_resetuj,
                            tree_action=nuluj_a_resetuj,
                            measurement_action=nuluj_a_draw_images
                            )
                # s.ImageSizes()
                with solara.Card(title="Signal source choice"):
                    solara.ToggleButtonsSingle(value=data_source, values=data_sources, on_value=draw_images)
                with solara.Card(title="Manual signal end"):
                    solara.InputFloat(label="Signal end time",value=manual_signal_end, on_value=draw_images, optional=True)
                    solara.Text("Manuálně nastavený konec nebo None.")
                with solara.Card(title="Plot data"):
                    solara.Button("Signal", on_click=lambda: create_overlay_signal('signal'))                    
            try:
                damping_graphs()
            except:
                solara.Error("Some problem appeared")
        with solara.lab.Tab("From amplitudes (tree overview)"):
            with solara.Sidebar():
                s.Selection(exclude_M01=True,
                            optics_switch=False,
                            day_action=resetuj,
                            tree_action=resetuj,
                            )
                with solara.Card(title="Parameter"):
                    solara.ToggleButtonsSingle(value=damping_parameter, values = damping_parameters)

                with solara.Card(title="Selection type"):
                    solara.ToggleButtonsSingle(value=data_selection, values = data_selection_types)
                    solara.Markdown("""
                    * **all**: show all data
                    * **optimistic**: remove data marked as failed by all people
                    * **pesimistic**: remove data marked as failed by at least one person
                    """)

                with solara.Card(title="Bounds for the filter in the table"):
                    solara.InputFloat("Lower bound for R", filtr_R_min)
                    solara.InputFloat("Upper bound for R", filtr_R_max)
                    solara.InputFloat("Lower bound for t/T", filtr_T_min)
                    solara.InputFloat("Upper bound for t/T", filtr_T_max)
                    solara.Text("""
                    Here you can set the bounds for R and the ratio of the length of the signal and the period. Note that R is negative and should be close to -1 for a good match.
                    The values which do not fulfill the filter conditions are replaced by nan values. 
                    """)
                with solara.Card(title="Popis"):
                    solara.Markdown(f"""
                    * Zelené dny odpovídají olistěnému stavu
                    * Přepnutím stromu se aktualizuje tabulka
                    * Nastavením ostatních parametrů a přepnutím na předchozí panel je možné zobrazit 
                      kmity, proložení nebo celý experiment.    
                    """)
            show_data_one_tree()
        with solara.lab.Tab("Probes comparison"):
            with solara.Sidebar():
                s.Selection(exclude_M01=True,
                            optics_switch=False,
                            day_action=resetuj,
                            tree_action=resetuj,
                            )

            probes_comparison()
    
        with solara.lab.Tab("Remarks"):
            remarks()
        # with solara.lab.Tab("From FFT (images)"):
        #     with solara.Sidebar():
        #         s.Selection(exclude_M01=True,
        #                     optics_switch=False,
        #                     day_action=resetuj2,
        #                     tree_action=resetuj2,
        #                     measurement_action= do_find_peaks
        #                     )
        #         # s.ImageSizes()
        #     try:
        #         peak_width_graph()
        #     except:
        #         pass
        # with solara.lab.Tab("From FFT (tables)"):
        #     with solara.Sidebar():
        #         with solara.Card(title="Background gradient"):
        #             solara.ToggleButtonsSingle(value=gradient_axis, values=gradient_axes)
        #     # try:
        #     peak_width_table()
        # # except:
        # #     solara.Error("Some problem appeared")


def probes_comparison():
#     with solara.Info():
#         solara.Markdown("""
# * The comparison of sensors for one measurement.
#         """, style={'color': 'inherit'})

    def process_row(row):
        color = ["red", "blue", "black", "green", "violet", "purple", "yellow"]
        data = {}
        fig = go.Figure()
        m = DynatreeMeasurement(day=row['date'],
                                tree=row['tree'],
                                measurement=row['measurement'],
                                measurement_type=row['type'])
        for i,source in enumerate(["Pt3", "Pt4", "Elasto(90)", "blueMaj", "yellowMaj"]):
            try:
                s = DynatreeDampedSignal(measurement=m, signal_source=source,  # dt=0.0002,
                                            # damped_start_time=54
                                            )
                ans = s.ldd_from_two_amplitudes(max_n=None)
                data[(*row.values(), source)] = [ans[i] for i in ["LDD", "R", "n", "peaks", "LDD_ans"]]
                scaling = np.max(np.abs(s.damped_signal))
                fig.add_trace(go.Scatter(
                    x=s.damped_time,
                    y=s.damped_signal / scaling,
                    mode='lines',
                    name=source,
                    line=dict(width=1.5, color=color[i]),
                    hovertemplate=f"{source}: %{{y:.2f}}<extra></extra>", 
                    legendgroup=source, 
                    showlegend=True
                ))
                fig.add_trace(go.Scatter(
                    x=ans['peaks'].index,
                    y=ans['peaks'] / scaling,
                    mode='markers',
                    marker=dict(size=8, color=color[i]),  # velikost a barva markerů
                    legendgroup=source,
                    showlegend=False  # skryje z legendy
                ))
                # print(ans['peaks'])
            except Exception as e:
                print(f"Error processing {row['date']} {row['tree']} {row['measurement']} {source}: {e}")
                data[(*row.values(), source)] = [None] * 5
        df = pd.DataFrame.from_dict(data, orient='index')
        df = df.reset_index()
        df.columns = ['experiment', 'LDD', 'R', 'n', 'peaks', 'LDD_ans']
        return {'data':df, 'figure':fig}

    def refactor_df(ans):
        data = ans.experiment.apply(pd.Series)
        data = data.rename(columns={0: "day", 1: "tree", 2: "measurement", 3: "type", 4: "source"})
        # Now we can concatenate the data with the rest of the columns
        data = pd.concat([data, ans.drop(columns=['experiment'])], axis=1)
        return data

    ans = process_row({'date': s.day.value, 'tree': s.tree.value, 'measurement': s.measurement.value, 'type': s.method.value})
    df = refactor_df(ans['data'])
    solara.display(df)
    with solara.Row():
        solara.FigurePlotly(ans['figure'])
    df = df[["source","LDD_ans"]]
    df_exploded = (
        df.explode('LDD_ans', ignore_index=True)
        .rename(columns={"LDD_ans": "LDD"})
        )

    fig = px.box(df_exploded, x='source', y='LDD', points='all', title='Boxplot according to source')
    solara.FigurePlotly(fig)

    df = pd.read_csv(config.file['outputs/damping_comparison'], index_col=None)
    df_wide = df.pivot_table(index=['day', 'tree', 'measurement', 'type'],
                            columns='source',
                            values=['LDD']).reset_index()
    
    with solara.Card(title=f"LDD for {s.tree.value} {s.day.value} {s.method.value}"):
        subdf = df_wide[(df_wide.tree==s.tree.value) & (df_wide.day==s.day.value)  & (df_wide.type==s.method.value)]
        subdf = subdf.set_index(['day', 'tree', 'measurement', 'type']).reorder_levels([1, 0, 3, 2]).sort_index()
        solara.display(subdf.style.background_gradient())

    with solara.Card(title=f"LDD for tree {s.tree.value} and all datasets"):
        subdf = df_wide[df_wide.tree==s.tree.value]
        subdf = subdf.set_index(['day', 'tree', 'measurement', 'type']).reorder_levels([1, 0, 3, 2]).sort_index()
        solara.display(subdf.style.background_gradient())

    solara.Markdown("```\n"+show_load_code(s)+"\n```")

    html_and_java_for_images()
    with solara.Sidebar():
        with solara.Card():        
            solara.Markdown("### Links")
            for i in ["Pt3", "Pt4", "Elasto(90)", "blueMaj", "yellowMaj"]:
                solara.HTML(unsafe_innerHTML=f"""
                    Data {i}:
    <a
    href="https://euler.mendelu.cz/fast/index.html?method={s.day.value}_{s.method.value}&tree={s.tree.value}&measurement={s.measurement.value}&sensor={i}&start=0&end=1000000000&format=html",
    target="_"
    class="image-preview" 
    data-src='https://euler.mendelu.cz/draw_graph/?method={s.day.value}_{s.method.value}&tree={s.tree.value}&measurement={s.measurement.value}&probe={i}&format=png'        
    >Full graph</a>
    or 
    <a
    href="https://euler.mendelu.cz/draw_graph_damping/?method={s.day.value}_{s.method.value}&tree={s.tree.value}&measurement={s.measurement.value}&probe={i}&format=png&damping_method=defmulti",
    target="_"
    class="image-preview" 
    data-src='https://euler.mendelu.cz/draw_graph_damping/?method={s.day.value}_{s.method.value}&tree={s.tree.value}&measurement={s.measurement.value}&probe={i}&format=png&damping_method=defmulti'        
    >Damped part</a>
    """)

# current = {'from_amplitudes': None, 'from_fft': None}
# gradient_axis = solara.reactive("Columns")
# gradient_axes = ["Rows", "Columns", "Table"]
# @solara.component
# def peak_width_table():
#     df = pd.read_csv(config.file['outputs/peak_width'])
#     df = df.pivot(index=df.columns[:4], columns="probe", values="width").drop(["a04_y", "a04_z"], axis=1)
#     trees = df.index.get_level_values('tree').drop_duplicates()


#     if gradient_axis.value == "Rows":
#         axis = 1
#     elif gradient_axis.value == "Columns":
#         axis = 0
#     else:
#         axis = None
#     for tree in trees:
#         _ = (
#             df[df.index.get_level_values('tree') == tree]
#             .style.format(precision=3).background_gradient(axis=axis)
#             .map(lambda x: 'color: lightgray' if pd.isnull(x) else '')
#             .map(lambda x: 'background: transparent' if pd.isnull(x) else '')
#         )
#         with solara.Card(title=f"Tree {tree}"):
#             solara.display(_)

# @solara.component
# def peak_width_graph():
#     global current
#     with solara.Sidebar():
#         with solara.Column():
#             solara.Button("Plot/Replot", on_click=do_find_peaks, color='primary')
#     coords = [s.tree.value, s.day.value, s.method.value, s.measurement.value]
#     if current['from_fft'] != coords:
#         current['from_fft'] = coords
#         do_find_peaks()
#     solara.Markdown(f"## {' '.join(coords)}")
#     solara.Info(
#         f"Relative peak width (peak width at given height divided by the peak position). Click Plot/Replot for another measurement. It takes few seconds to draw all sensors.")
#     solara.ProgressLinear(find_peak_widths.pending)

#     if not find_peak_widths.finished:
#         return
#     with solara.Row(style={'flex-wrap': 'wrap'}):
#         for target_probe, ans in zip(data_sources, find_peak_widths.value):
#             coordsf = [s.method.value, s.day.value, s.tree.value, s.measurement.value, target_probe]
#             if ans is None:
#                 continue
#             with solara.Card(title=f"{target_probe}: {round(ans['width'], 4)}", style={'min-width': '150px'}):
#                 if coordsf in df_failed_FFT_experiments.values.tolist():
#                     solara.Error("This measurement has been marked as failed.")
#                 try:
#                     solara.FigureMatplotlib(ans['fig'])
#                 except:
#                     pass

#                 def create_button(label, target_probe, output):
#                     # Funkce vytvoří tlačítko a uzavře aktuální hodnotu target_probe
#                     return solara.Button(label=label, on_click=lambda x=None: open_dialog(target_probe, output=output))

#                 # Vytvoření tlačítek
#                 create_button("Show experiment", target_probe, output='experiment')
#                 create_button("Show signal for FFT", target_probe, output='signal')
#                 create_button("Show FFT spectrum", target_probe, output='fft')

#                 # solara.Button(label="Show signal", on_click=lambda x=None: open_dialog(target_probe, output='signal'))
#                 # solara.Button(label="Show FFT spectrum",
#                 #               on_click=lambda x=None: open_dialog(target_probe, output='fft'))
#     plt.close('all')
#     create_overlay()


# @solara.component
# def create_overlay():
#     with ConfirmationDialog(show_dialog.value, on_ok=close_dialog, on_cancel=close_dialog, max_width='90%',
#                             title=""):
#         if open_dialog.finished:
#             ans = open_dialog.value
#             solara.Markdown(f"## {' '.join(ans['coords'])}")
#             ans['data'].name = "value"
#             fig = ans['data'].plot(backend='plotly')
#             if ans['output'] == 'fft':
#                 fig.update_layout(
#                     yaxis=dict(type="log"),
#                     xaxis=dict(range=[0, 10]),
#                     height=500,
#                     width=700,
#                 )
#             solara.FigurePlotly(fig)


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

    coordsf = [s.method.value, s.day.value, s.tree.value, s.measurement.value, data_source.value]
    if coordsf in df_failed_FFT_experiments.values.tolist():
        solara.Error("This measurement has been manually marked as failed in the file csv/FFT_failed.csv.")
        return None

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
    solara.Markdown(f"## {' '.join(coords)}")
    # if current['from_amplitudes'] != coords:
    #     current['from_amplitudes'] = coords
    #     dynatree.logger.info("Graphs from peaks are not current. Calling draw_images.")
    #     draw_images()
    #     # return
    if not draw_images.finished:
        solara.Error("""
        Probíhá výpočet. Trvá řádově vteřiny.Pokud se tato zpráva zobrazuje déle, něco je špatně. 
        Možná je vybrána kombinace vstupů, která není zpracována.  
        V takovém případě vyber jiný druh měření, den, strom nebo jiný probe.
        """)
        solara.SpinnerSolara(size="100px")
        return
    ans = draw_images.value
    if ans is None:
        solara.Error("Nekde nastala chyba")
    else:
        df = ans['df']
        df = pd.concat([df.loc[['LDD']], df.drop(index='LDD')])
        fig = ans['fig']
        marked_failed = ans['failed']
        background_color = 'transparent'
        T = 1 / ans['peak']
        interval_length = ans['signal_peaks'].index[-1]-ans['signal_peaks'].index[0]
        err_info = ""
        if marked_failed == True:
            err_info = f"{err_info} This measurement was marked as failed."
        if interval_length/T<2:
            err_info = f"{err_info} The interval is shorter than the double of the period."
        if df.loc["R"].max()>-0.9:
            err_info = f"{err_info} Some of the R is outside the interval (-1,-0.9)."
        if err_info:
            solara.Error(solara.Markdown(f"""
            {err_info} You may want to add the following line to the file `csv/damping_failed.csv`
            ```
            {s.method.value},{s.day.value},{s.tree.value},{s.measurement.value},{data_source.value}
            ```
            """,
            style = {'color':'inherit'}))
            # background_color = '#f8d7da'
        with solara.Card(style={'background-color': background_color}):
            with solara.Row():
                # df.loc["LDD",:] = df.loc['b',:] * T

                # Vlastní funkce pro formátování
                def custom_format(x):
                    if abs(x) < 1e-5 and x != 0:  # Pokud je číslo menší než 1e-5 (kromě nuly), zobrazí se ve vědeckém formátu
                        return f"{x:.2e}"
                    else:  # Jinak zobrazí 8 desetinných míst
                        return f"{x:.8f}"
                solara.display(df.style.format(custom_format).background_gradient(axis=1))
                with solara.Column():
                    solara.Markdown("The signal envelope is $e^{-bt}$.")
                    solara.Text(f"Main freq is f={ans['peak']:.5} Hz, period is T={T:.5} s.")
                    solara.Text("LDD = b*T")
                    solara.Text(f"The length of the analyzed time interval is {interval_length:.2f}.")
                    solara.Text(f"The length of the analyzed time interval is {interval_length/T:.2f} times the period.")
                    df = pd.read_csv(config.file['damping_manual_ends'], skipinitialspace=True)
                    df = df.loc[(df["tree"]==s.tree.value) & (df["measurement"]==s.measurement.value)
                                & (df["day"] == s.day.value) & (df["measurement_type"] == s.method.value)]
                    if len(df) > 0:
                        solara.Markdown(f"**Suggested end time for damping extraction is {df.iat[0,-1]}.**")
                    else:
                        solara.Text(f"The signal is analyzed up to the end (no manual end time).")

            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',  # Pozadí celého plátna
                              #plot_bgcolor='rgba(0,0,0,0)'
                              )
            solara.FigurePlotly(fig)
        with solara.Warning():
            solara.Markdown("""
            TODO: 

            * Nepoužívat příliš dlouhý časový interval. Konec nastavit na 15% maxima. Zatím je nastaveno 
              u metody využívající extrémy. Hilbert a wavelety tuto informaci přebírají. Je to tak dostatečné?
            * Možná bude potřeba opravit hledání peaků a další parametry pro optiku a akcelerometry.   
            * Možná bude potřeba doladit vycentrování signálu tak, aby hilbert měl co nejmenší zvlnění. UPDATE: pokus byl. moc to vliv nemělo     
            """, style={'color': 'inherit'})
        with solara.Info():
            solara.Markdown(
                """
                * **Extrémy**: 
                    * Po vypuštění se vynechá půlperioda. 
                    * Peaky nesmí být blíže než 75 procent periody.
                    * Po prvním peaku, který je pod 15 procent maxima se signál už neuvažuje.
                * **Hilbert**
                    * Analyzovaný časový úsek stejný jako u metody extrémů.
                    * Zvážit, jestli by se nedalo zvlnění ovlivnit odstraněním trendu.
                * **Wavelet**
                    * Analyzovaný časový úsek stejný jako u metody extrémů.
                * **Definice**
                    * Vzorec $$\\mathrm{LDD} = \\ln \\frac{y_0}{y_2},$$ kde $y_0$ a $y_2$ jsou amplitudy ve dvou po sobě jdoucích
                      maximech nebo minimech. Ze všech hodnot se bere medián. Variabilitu popisuje směrodatná odchlka, ale ta není 
                      porovnávatelná s metodami založenými na obálkách.
                * **Definice s více amplitudami**
                    * Snaží se odfiltrovat nežádoucí vliv toho, že střední hodnota může růst, dokonce nelineárně. Proto nepracuje s amplitudami 
                      (vzdálenost peaku od střední hodnoty), ale s vertikální vzdáleností po sobě následujících peaků. Je možné mít víc variant. 
                      Jako první nástřel je použita metoda využívající dvě po sobě jdoucí maxima a minimum mezi nimi, nebo naopak. 
                      $$\\mathrm{LDD} = 2 \\ln \\frac{|y_0-y_1|}{|-y_1+y_2|}$$
                * V tabulce jsou i další metriky z lineární regrese, ale hilbertova obálka i wavelet používají 
                  řádově jiný počet bodů a proto není možné srovnávat například p-hodnoty.
                """, style={'color':'inherit'}
            )

def nuluj_a_draw_images(x=None):
    manual_signal_end.value = None
    draw_images()

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
    sig = DynatreeDampedSignal(m, data_source.value, dt=dt, damped_end_time=manual_signal_end.value)

    data = {}
    keys = ['b', 'R', 'p', 'std_err', 'LDD']
    fig = make_subplots(rows=3, cols=1, shared_xaxes='all', shared_yaxes='all')

    ans = sig.hilbert_envelope
    fig = draw_signal_with_envelope(sig, fig, ans['data'], ans['b'], ans['q'], row=1)
    data['hilbert'] = [ans[key] for key in keys]

    ans = sig.fit_maxima()
    signal_peaks = ans['peaks']
    fig = draw_signal_with_envelope(sig, fig, k=ans['b'], q=ans['q'], row=2)
    fig.add_trace(go.Scatter(x=signal_peaks.index, y=signal_peaks.values.reshape(-1),
                             mode='markers', name='peaks', line=dict(color='red')), row=2, col=1)
    data['extrema'] = [ans[key] for key in keys]

    ans = sig.wavelet_envelope
    fig = draw_signal_with_envelope(sig, fig, ans['data'], k=ans['b'], q=ans['q'], row=3)
    data['wavelets'] = [ans[key] for key in keys]

    fig.update_layout(title=f"{m}", height=800)

    fig.update_yaxes(title_text="Hilbert", row=1, col=1)
    fig.update_yaxes(title_text="Maxima/minima", row=2, col=1)
    fig.update_yaxes(title_text="Wavelet", row=3, col=1)

    df = pd.DataFrame.from_dict(data)
    df.index = keys
    df.loc["T",:] = 1/sig.main_peak
    temp = sig.ldd_from_definition()
    df.loc[["b","LDD","T", "std_err"], "def"] = [temp['b'], temp['LDD'], temp['T'], temp["std_err"]]
    temp = sig.ldd_from_definition(peaks_limit=3)
    df.loc[["b","LDD","T", "std_err"], "def2"] = [temp['b'], temp['LDD'], temp['T'], temp["std_err"]]
    temp = sig.ldd_from_distances()
    df.loc[["b","LDD","T", "std_err"], "def2diff"] = [temp['b'], temp['LDD'], temp['T'], temp["std_err"]]
    temp = sig.ldd_from_two_amplitudes()
    cols = ["b","LDD","T", "std_err", "R"]
    df.loc[cols, "defmulti"] = [temp[i] for i in cols]
    df.loc["n","defmulti"] = temp['n']

    return {'df':df, 'fig':fig, 'failed':sig.marked_failed, 'peak':sig.main_peak,
            'signal_peaks':signal_peaks
            }


def html_and_java_for_images():
    solara.Style(".level1 {border-style: solid !important; border-color:gray !important; border-width:1px !important;}")
    solara.HTML(tag="div", unsafe_innerHTML="""
    <div id="preview-container" class="preview-container">
    <img id="preview-image" src="" alt="Náhled">
    <div id="preview-comment">AAAA</div>
    </div>
    <button id="startBtn" class="v-btn v-btn--contained theme--light v-size--default" style="display:none">Začni sledovat pohyb myši</button>
    """)
    solara.HTML(tag="script", unsafe_innerHTML="""
function initPreview() {
const previewContainer = document.getElementById("preview-container");
const previewImage = document.getElementById("preview-image");
const previewComment = document.getElementById("preview-comment");    
const links = document.querySelectorAll(".image-preview");
links.forEach(link => {
    if (!link.dataset.listenerAdded) { // Zabrání opakovanému přidání posluchačů
        link.addEventListener("mouseenter", function () {
            const imageUrl = this.getAttribute("data-src");
            previewImage.src = imageUrl;
            previewContainer.style.display = "block";
            // Zpracování textového náhledu
            const textSrc = this.getAttribute("data-text-src");
            if (textSrc) {
                fetch(textSrc)
                    .then(response => {
                        console.log("Status odpovědi:", response.status);
                        console.log("Content-Type:", response.headers.get("Content-Type"));
                        if (!response.ok) throw new Error("Chyba při načítání JSON");
                        return response.json();
                    })
                    .then(data => {
                        console.log("Načtená data:", data);
                        
                        if (Array.isArray(data.comments)) {
                            // Spojení všech textů do jednoho stringu
                            // const commentText = data.comments.map(comment => comment.text).join(" | ");
                            // previewComment.textContent = commentText || "Žádné komentáře";
                            previewComment.innerHTML = createTable(data.comments);
                        } else {
                            previewComment.textContent = "Neplatná data";
                        }
                        //previewComment.textContent = data.comments || "Žádný komentář";
                    })
                    .catch(error => {
                        console.error("Nepodařilo se načíst komentář:", error);
                        previewComment.textContent = "Chyba při načítání komentáře";
                    });
            } else {
                previewComment.textContent = ""; // Vyčistit obsah, pokud není `data-text-src`
            }
        });
        link.addEventListener("mouseleave", function () {
            previewContainer.style.display = "none";
        });

        link.style.color = "green";
        link.dataset.listenerAdded = "true"; // Označíme, že posluchač už byl přidán
        console.log("Listener přidán k odkazu");
    }
});
}

// Spuštění kódu po kliknutí na tlačítko
document.getElementById('startBtn').addEventListener('click', initPreview);

// Automatické sledování změn v DOM pro nové prvky
const observer = new MutationObserver(() => {
initPreview();
});

// Sledování změn v celém dokumentu (můžeš omezit na konkrétní kontejner)
observer.observe(document.body, { childList: true, subtree: true });

// První inicializace sekundu po načtení stránky
window.addEventListener('load', function() {
setTimeout(initPreview, 1000);
});

// Funkce pro generování tabulky z JSON dat
function createTable(comments) {
let table = `<ul>`;

comments.forEach(comment => {
    table += `<li>
        <b>${comment.text}</b>
        <b>(Hodnocení ${comment.rating})</b>
    </li>`;
});

table += `</ul>`;
return table;
}
    """)

@solara.component
def show_data_one_tree():
    list_of_methods = ['maxima', 'hilbert', 'wavelet']
    with solara.Card():
        solara.Markdown(f"## {s.tree.value}")

        solara.ToggleButtonsSingle(value=switch_damping, values=["defmulti","all methods"])

        df = pd.read_csv(config.file['outputs/damping_factor'])
        df_matlab = (pd.read_csv("../data/matlab/utlum_FFT_3citlivost_15amp.csv")
                     .rename(columns = {'folder':'day', 'b':'FFT_b', 'LDD':'FFT_LDD'})
                     )
        df_matlab["FFT_R"] = np.sqrt(df_matlab.R2)
        df_matlab[['type', 'tree', 'measurement']] = df_matlab['Name'].str.split('_', expand=True)
        df_matlab['probe'] = "Elasto(90)"
        df_matlab['day'] = df_matlab['day'].str.replace('_', '-')

        df_matlab = (df_matlab
                        .loc[:, ["day", "type", "tree", "measurement", "probe", "FFT_b", "FFT_R", "FFT_LDD"]]
                        .dropna()
                     )
        df = df.merge(df_matlab, how='left')
        # Nahradí hodnoty ve sloupcích bez "_R" None pokud odpovídající "_R" sloupec má hodnotu > -0.9
        # for col in list_of_methods:
        #     df.loc[df[f"{col}_R"] > -0.9, f"{col}_R"] = None

        df = df[df["tree"]==s.tree.value]
        df["#_of_periods"] = (df["end"]-df["start"]) * df["freq"]

        type_order = ['normal', 'noc', 'den', 'afterro', 'afterro2', 'mraz', 'mokro']
        df['type'] = pd.Categorical(df['type'], categories=type_order, ordered=True)
        df["linkPNG"] = df.apply(lambda row:f"""
        <a  href='https://euler.mendelu.cz/draw_graph/?method={row['day']}_{row['type']}&tree={row['tree']}&measurement={row['measurement']}&probe=Elasto%2890%29&start=0&end=1000000000&format=png'
        class="image-preview" 
        data-src='https://euler.mendelu.cz/draw_graph/?method={row['day']}_{row['type']}&tree={row['tree']}&measurement={row['measurement']}&probe=Elasto%2890%29&start=0&end=1000000000&format=png'
        data-text-src='https://euler.mendelu.cz/gallery/api/comments/utlum/{row['day']}_{row['type']}_{row['tree']}_{row['measurement']}.png'        
        >PNG</a>"""
                                 , axis=1)

        if switch_damping.value != 'defmulti':
            df["links"] = df.apply(lambda row:f"""
        <a  href='https://euler.mendelu.cz/draw_graph_damping/?method={row['day']}_{row['type']}&tree={row['tree']}&measurement={row['measurement']}&probe=Elasto%2890%29&format=png&damping_method=extrema'
        class="image-preview" 
        data-src='https://euler.mendelu.cz/draw_graph_damping/?method={row['day']}_{row['type']}&tree={row['tree']}&measurement={row['measurement']}&probe=Elasto%2890%29&format=png&damping_method=extrema'        
        >M</a>
        <a  href='https://euler.mendelu.cz/draw_graph_damping/?method={row['day']}_{row['type']}&tree={row['tree']}&measurement={row['measurement']}&probe=Elasto%2890%29&format=png&damping_method=hilbert'
        class="image-preview" 
        data-src='https://euler.mendelu.cz/draw_graph_damping/?method={row['day']}_{row['type']}&tree={row['tree']}&measurement={row['measurement']}&probe=Elasto%2890%29&format=png&damping_method=hilbert'        
        >H</a>
        <a  href='https://euler.mendelu.cz/draw_graph_damping/?method={row['day']}_{row['type']}&tree={row['tree']}&measurement={row['measurement']}&probe=Elasto%2890%29&format=png&damping_method=wavelet'
        class="image-preview" 
        data-src='https://euler.mendelu.cz/draw_graph_damping/?method={row['day']}_{row['type']}&tree={row['tree']}&measurement={row['measurement']}&probe=Elasto%2890%29&format=png&damping_method=wavelet'        
        >W</a>
        <a  href='https://euler.mendelu.cz/draw_graph_damping/?method={row['day']}_{row['type']}&tree={row['tree']}&measurement={row['measurement']}&probe=Elasto%2890%29&format=png&damping_method=defmulti'
        class="image-preview" 
        data-src='https://euler.mendelu.cz/draw_graph_damping/?method={row['day']}_{row['type']}&tree={row['tree']}&measurement={row['measurement']}&probe=Elasto%2890%29&format=png&damping_method=defmulti'        
        >DefMulti</a>
        <a  href='https://euler.mendelu.cz/gallery/api/comments/utlum/{row['day']}_{row['type']}_{row['tree']}_{row['measurement']}.png'        
        >Comments</a>
"""
                                 , axis=1)
        else:
            df["links"] = df.apply(lambda row: f"""
                    <a  href='https://euler.mendelu.cz/draw_graph_damping/?method={row['day']}_{row['type']}&tree={row['tree']}&measurement={row['measurement']}&probe=Elasto%2890%29&format=png&damping_method=defmulti'
                    class="image-preview" 
                    data-src='https://euler.mendelu.cz/draw_graph_damping/?method={row['day']}_{row['type']}&tree={row['tree']}&measurement={row['measurement']}&probe=Elasto%2890%29&format=png&damping_method=defmulti'        
                    >DefMultiEnvelope</a>
                    <a  href='https://euler.mendelu.cz/gallery/api/comments/utlum/{row['day']}_{row['type']}_{row['tree']}_{row['measurement']}.png'        
                    >Comments</a>
            """
                                   , axis=1)

        df["linkHTML"] = df.apply(lambda row:f"<a href='https://euler.mendelu.cz/fast/index.html?method={row['day']}_{row['type']}&tree={row['tree']}&measurement={row['measurement']}&sensor=Elasto%2890%29&start=0&end=1000000000&format=html '>html</a>", axis=1)
        df = df.set_index(["tree","day", "type", "measurement"])
        df = df.sort_index()

        cols = [i for i in df.columns if "_" in i]
        df.loc[~((df["#_of_periods"] > filtr_T_min.value) & (df["#_of_periods"] < filtr_T_max.value)), cols] = np.nan
        for i in list_of_methods:
            df.loc[
                ~((df[f"{i}_R"] > filtr_R_min.value) & (df[f"{i}_R"] < filtr_R_max.value)), [f"{i}_b", f"{i}_LDD"]] = np.nan
        # df[~ ((df["#_of_periods"] > filtr_T_min.value) & (df["#_of_periods"] < filtr_T_max.value)),:] = np.nan

        if data_selection.value != "all":
            if data_selection.value == "optimistic":
                key = "max"
            else:
                key = "min"
            failed = get_zero_rating(key=key, tree=s.tree.value)
            failed = [tuple(i) for i in failed[["tree","day", "type", "measurement"]].values]
            # breakpoint()
            # print (df.index.isin(failed))
            df.loc[df.index.isin(failed), cols] = np.nan

        cols = [i for i in df.columns if f"_{damping_parameter.value}" in i] + ["defmulti_R",
            "defmulti_n","linkPNG", "links", "linkHTML"]
        df = df[cols]
        df = df.sort_index()

        # Definujeme styly pro index
        def highlight_index(val):
            if val in config.summer_dates:
                return "background-color: lightgreen; font-weight: bold;"
            return ""

        if switch_damping.value != 'defmulti':
            df = df.drop(["defmulti_R", "defmulti_n"], axis=1)
            background_axis = None
            comment = """
            * All methods are shown.
            * The most suitable (as of May 2025) is "defmulti" whic is evaluated from modified definition for three 
              signal values and involve the differences rather than signal values.
            * To see aditional info about defmulti, switch the buttons above to "defmulti". 
            """
        else:
            df = df[["defmulti_LDD", "defmulti_R", "defmulti_n", "linkPNG", "links", "linkHTML"]]
            background_axis = 0
            df.loc[df["defmulti_n"].isna(),"defmulti_n"] = 0
            df["defmulti_n"] = df["defmulti_n"].astype(int)
            comment = """
            * The data are calculated from modified definition for multiple (3) signal values and involve the differences rather than signal values.
            * The R coefficient is a measure, if the approximation is good or bad. (Dark background means worse approximation and less reliable damping value.)
            * The n coefficient show, how many point triples are included in the computation. (Dark background means more points and more reliable damping value.)
            """

        solara.Info(solara.Markdown(comment, style='inherit'))
        _ = (
            df
            .style.format(precision=3).background_gradient(axis=background_axis)
            .apply( lambda x:add_horizontal_line(x, second_level=False, level0=1), axis = None)
            .map_index(highlight_index, level=1)  # Obarví druhou úroveň indexu
            .map(lambda x: 'color: lightgray' if pd.isnull(x) else '')
            .map(lambda x: 'background: transparent' if pd.isnull(x) else '')
        )

        html_and_java_for_images()

        display(_)


        if data_selection.value != 'all':
            with solara.Card(title = "Data manually marked as failed"):
                solara.Markdown("""
                The following data are marked as failed by setting the worst rating (only 1 star) in 
                the [image gallery](https://euler.mendelu.cz/gallery/gallery/utlum).
                """)
                display(failed)

        with solara.Warning():
            solara.Markdown("""
            * **maxima_LDD**  - stanoveno ze všech maxim a minim prokládáním exponenciely.
            * **hilbert_LDD**  - stanoveno z hilbertovy obálky prokládáním exponenciely.
            * **wavelet_LDD**  - stanoveno z hilbertovy obálky prokládáním exponenciely.
            * **def_LDD** - stanoveno z definice útlumu (výška peaku), medián za celý zpracovávaný signál.
            * **def2_LDD** - stanoveno z definice útlumu (výška peaku), první dvě maxima nebo první dvě minima, dle toho, co je dřív. *Pracuje jenom se začátkem.*
            * **def2diff_LDD** - stanoveno stejně jako předchozí, ale místo výšky peaku se bere vzdálenost mezi po sobě jdoucím maximem a minimem. Pracuje s prvními čtyřmi peaky. *Mělo by být robustní vůči svislému posunu.* *Pracuje jenom se začátkem.*   
            * **defmulti_LDD** - stanoveno stejně jako předchozí, ale bere se pro každou trojici peaku svislá vzdálenost mezi prostředním peakem a krajiními peaky. Potom se bere medián na celém signálu. *Mělo by být robustní vůči svislému posunu.* 
            * **FFT_LDD** - Patrikova DownHill metoda. *Mělo by být robustní vůči svislému posunu.*            
            """, style={'color':'inherit'})

        with solara.Info():
            solara.Markdown("""
* Nejprve se určí studovaný interval, podle prvního nulového bodu za vypuštěním.  Poté se hledají peaky (maxima a minima).
    * Při hledání peaků se začne půl periody za začátkem, aby se odfiltroval první peak ovlivněný vypuštěním.
    * Poté se hledají peaky na absolutní hodnotě signálu tak, aby vzdálenost byla minimálně 75 procent půlperiody.
    * S hledáním peaků se končí když první peak klesne pod 15 procent maxima, nastaveno v proměnné `config.damping_threshold`.
* Nalezené peaky se použijí pro hledání fitovnám exponenciely (maxima_LDD), stanovením mediánu podílů amplitud (def_LDD), stanovením 
    mediánů rozdílů amplitud pro tři po sobě jdoucí body (defmulti_LDD).
* Dále se nalezené peaky použijí pro stanovení intervalu pro wavelet a hilbertovu obálku.
* První dva kladné peaky nebo první dva záporné peaky se použijí pro stanovení LDD ze dvou po sobě jdoucích amplitud. Zdá se to být citlivé na 
  případné asymetrie v signálu.   
* Uřiznutí konce má vliv na centrování pomocí střední hodnoty, pokud na konci jsou funkční hodnoty výrazně kladné nebo záporné.
  Pokud signál osciluje okolo nuly, tak se ořiznutí udělá automaticky na 15 procentech maxima.               
            """, style={'color':'inherit'})
@solara.component
def remarks():
    with solara.Sidebar():
        with solara.Info():
            solara.Text("Seznam měření, kde něco selhalo. Buď zpracování nebo experiment.")
    df = pd.DataFrame([
['2022-04-05 normal B16 M03','Rozsypany caj'],
['2022-08-16 noc BK08 M05','Rozsypany caj'],
['2024-09-02 mokro BK08 M04','Neni tak spatny ale neco se stalo a skript chcipnul. Urezat konec?'],
['2024-04-10 normal BK16 M02','Nejsou data'],
['2022-08-16 noc BK08 M04','Nejsou data'],
['2024-09-02 mokro BK07 M02','Nejsou data'],
['2024-09-02 mokro BK08 M03','Neni v tabulce, nevim proc skript chcipnul. Prozkoumat.'],
['2023-07-17 normal BK16 M04','Bylo vyhozeno z FFT analyz a tim i odsud. Ale snad by slo zachranit vyberem intervalu.|']
], columns=['dataset','problem'])
    # solara.HTML(unsafe_innerHTML=df.to_html())
    with solara.Info():
        solara.Markdown("""
Některá měření nejsou vyhodnocena.    
    
|dataset|co se stalo|
|--|--|        
|2022-04-05 normal B16 M03|Rozsypany caj|
|2022-08-16 noc BK08 M05|Rozsypany caj|
|2024-09-02 mokro BK08 M04|Neni tak spatny ale neco se stalo a skript chcipnul. Urezat konec?|
|2024-04-10 normal BK16 M02|Nejsou data|
|2022-08-16 noc BK08 M04|Nejsou data|
|2024-09-02 mokro BK07 M02|Nejsou data|
|2024-09-02 mokro BK08 M03|Neni v tabulce, nevim proc skript chcipnul. Prozkoumat.|
|2023-07-17 normal BK16	M04|Bylo vyhozeno z FFT analyz a tim i odsud. Ale snad by slo zachranit vyberem intervalu.|
    """, style={'color': 'inherit'})
        # display(df)
    with solara.Info():
        solara.Markdown(
            """
        * Některá měření se nezpracovávala, byla vyhozena jako špatná během analýzy FFT. Jsou zapsána v csv souboru 
        [FFT_failed.csv](https://github.com/arborist-mendelu/dynatree/blob/master/scripts/csv/FFT_failed.csv).
        * Soubor obsahuje všechny proby, zde je subset pro Elasto(90).
                        """,
                        style={'color': 'inherit'})
        df_failed = pd.read_csv(config.file['FFT_failed'])
        df_failed = df_failed[df_failed["probe"]=="Elasto(90)"].sort_values(by=["tree","day","type","measurement"])
        #https://euler.mendelu.cz/api/draw_graph/?method=2021-03-22_normal&tree=BK04&measurement=M02&probe=Elasto%2890%29&format=png

        df_failed["links"] = df_failed.apply(lambda row: f"""
                <a  href='https://euler.mendelu.cz/draw_graph/?method={row['day']}_{row['type']}&tree={row['tree']}&measurement={row['measurement']}&probe=Elasto%2890%29&format=png'
                >png image</a>
        """, axis=1)
        _ = df_failed.style.format(precision=3)

        display(_)
        # solara.HTML(unsafe_innerHTML=df_failed.to_html())

dynatree.logger.info(f"File damping.py loaded in {time.time() - loading_start} sec.")

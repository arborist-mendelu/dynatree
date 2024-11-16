from matplotlib.pyplot import axvline
import solara.lab
import solara.website
from dynatree import dynatree
import solara
import pandas as pd
# import polars as pl
import dynatree.solara.select_source as s
import matplotlib.pyplot as plt
from solara.lab import task
from dynatree.signal_knock import SignalTuk, find_peak_times_chanelA, find_peak_times_chanelB, chanelA, chanelB
from dynatree_summary.acc_knocks import  delta_time
import logging
import plotly.express as px
from io import BytesIO
import random
from contextlib import contextmanager
import time
from functools import lru_cache
import urllib.parse

dynatree.logger.setLevel(logging.INFO)
# dynatree.logger.setLevel(logging.ERROR)

dynatree.logger.info("Starting app tuk_ACC.py")
server = "http://um.mendelu.cz/dynatree/"
dynatree.logger.info(f"Server is {server}")
df = pd.read_csv("../outputs/FFT_acc_knock.csv")
if "valid" not in df.columns:
    df["valid"] = True
if "manual_peaks" not in df.columns:
    df["manual_peaks"] = None
rdf =  df.copy()
rdf['timecoords'] = rdf['day'].astype(str) + "_" + rdf['type']
df_updated = solara.reactive(0)
day_type_pairs = list(rdf.loc[:,["day","type"]].drop_duplicates().sort_values(by=["day","type"]).values)
day_type_pairs = [f"{i[0]}_{i[1]}" for i in day_type_pairs]
dynatree.logger.info("Dataframes initialized")


active_tab = solara.reactive(1)
use_overlay = solara.reactive(False)
manual_freq = solara.reactive([])
cards_on_page = solara.reactive(10)
use_large_fft = solara.reactive(False)
use_custom_file = solara.reactive(False)
img_from_live_data = solara.reactive(False)
img_size = solara.reactive('small')

filecontent = solara.reactive("")

@contextmanager
def customizable_card(overlay=True):
    if overlay:
        # První varianta s overlay efektem přes celou obrazovku
        with solara.Card(style={
            'position': 'fixed', 'top': '0', 'left': '0', 'width': '100vw', 'height': '100vh',
            'background-color': 'rgba(0, 0, 0, 0.5)', 'z-index': '999',
            'display': 'flex', 'align-items': 'center', 'justify-content': 'center'
        }, margin=0):
            with solara.Card(style={
                'background-color': 'white', 'padding': '20px', 'border-radius': '8px',
                'box-shadow': '0 4px 8px rgba(0, 0, 0, 0.2)',
                'max-width': '1200px', 'width': '1000px', 'max-height': '100vh'
            }):
                yield  # Výkon příkazu v této variantě
    else:
        # Druhá varianta, fixovaná v pravém dolním rohu
        with solara.Card(style={
            'position': 'fixed', 'bottom': '0px', 'right': '0px', 'z-index': '1000',
            'max-width': '1200px', 'width': '1000px', 'max-height': '100vh'
        }):
            yield  # Výkon příkazu v této variantě


def on_file(f):
    global rdf
    dynatree.logger.info("File uploaded")
    content = f["file_obj"].read()
    df = None
    try:
        dynatree.logger.info("Trying to read as csv ...")
        df = pd.read_csv(BytesIO(content))
        dynatree.logger.info("     Passed")
    except:
        dynatree.logger.info("     FAILED")
    if df is None:
        try:
            dynatree.logger.info("Trying to read as parquet ...")
            df = pd.read_parquet(BytesIO(content))
            dynatree.logger.info("     Passed")
        except:
            dynatree.logger.info("     FAILED")
    if df is not None:
        dynatree.logger.info(f"Head of uploaded data: {df.head()}")
    else:
        dynatree.logger.info(f"File not accepted")
    if df is not None:
        rdf = df.copy(deep=False)
        df_updated.value = random.random()

@solara.component
def Page():
    dynatree.logger.info("Page in tuk_ACC.py entered")
    router = solara.use_router()
    parsed_values = urllib.parse.parse_qs(router.search, keep_blank_values=True)
    dynatree.logger.info(f"Parsed values from URL: {parsed_values}")
    if 'active_tab' in parsed_values.keys():
        try:
            active_tab.value = int(parsed_values['active_tab'][0])
            router.push("ACC_tuk")
        except:
            pass
    solara.Title("DYNATREE: ACC ťuknutí")
    solara.Style(s.styles_css+".zero-margin p {margin-bottom: 0px;}")
    with solara.Sidebar():
        s.Selection(
            optics_switch=False,
            report_optics_availability=False,
            include_measurements=active_tab.value != 2
        )
        if active_tab.value == 1:
            with solara.Card(title = "Dashboard setting"):
                solara.Switch(label="Use overlay for graphs (useful for small screen)", value=use_overlay)
                solara.Switch(label="Use larger FFT image", value=use_large_fft)
                solara.Switch(label="FFT images from live peaks", value=img_from_live_data)
                with solara.Column(gap="0px"):
                    solara.Text("Items on page:")
                    solara.ToggleButtonsSingle(value=cards_on_page, values=[10,20,50,75,100])
            with solara.Card(title = "Data file handling"):
                with solara.Column():
                    solara.FileDownload(rdf.to_parquet(), filename="FFT_acc_knock.parquet",
                                        label="Download parquet data")
                    solara.FileDownload(rdf.to_csv(), filename="FFT_acc_knock.csv",
                                label="Download csv data")
                    solara.Switch(label="Allow file upload", value=use_custom_file)
                    if use_custom_file.value:
                        solara.FileDrop(
                                label="Drop your own csv or parquet file here.",
                                on_file=on_file,
                        )
    with solara.lab.Tabs(value=active_tab, lazy=True):
        with solara.lab.Tab("Měření"):
            Signal()
            Rozklad()
        # with solara.lab.Tab("Tabulka"):
        #     Tabulka()
        with solara.lab.Tab("Detail ťuků"):
            if use_overlay.value:
                Seznam()
                Graf()
            else:
                with solara.Columns(1,1):
                    with solara.Column():
                        Seznam()
                    with solara.Column():
                        Graf()
        with solara.lab.Tab("Tabulka pro strom a den"):
            Tabulka()
        with solara.lab.Tab("Jeden probe, všechna data"):
            Seznam_probe()

select_days_multi = solara.reactive([])
@solara.component
def Seznam_probe():
    with solara.Row():
        solara.ToggleButtonsMultiple(value=select_probe_multi, values = ["a01","a02","a03","a04"])
        solara.ToggleButtonsMultiple(value=select_axis_multi, values = ["x","y","z"])
        solara.ToggleButtonsSingle(value=img_size, values=["small","large"])
    solara.ToggleButtonsMultiple(value=select_days_multi, values=day_type_pairs)
    probeset = [f"{i}_{j}" for i in select_probe_multi.value for j in select_axis_multi.value]
    subdf = rdf[
        (rdf["tree"] == s.tree.value)
        &
        (rdf["probe"].isin(probeset))
    ]
    subdf = subdf[subdf['timecoords'].isin(select_days_multi.value)]
    subdf = subdf.sort_values(by=["day","type","measurement","knock_time"])
    sets = subdf[["day","type"]].drop_duplicates()
    file = f"<h1>Tree {s.tree.value} and probes {probeset}</h1>"
    images_added = False
    for I,R in sets.iterrows():
        subsubdf = subdf[(subdf["day"] == R["day"]) & (subdf["type"] == R["type"])]
        file = file + f"<h2> {R['day']} {R['type']} </h2>"
        for subprobe in probeset:
            sub3df = subsubdf[subsubdf["probe"] == subprobe]
            file = file + f"<h3>{subprobe}</h3>"
            for i, row in sub3df.iterrows():
                if img_size.value == 'small':
                    image_path = "/static/public/cache/FFT_" + row['filename'] + ".png"
                else:
                    image_path = "/static/public/fft_images_knocks/FFT_" + row['filename'] + ".png"
                souradnice = f"{row['measurement']} @{row['knock_time']/100.0}sec, <b>{round(row['freq'])} Hz</b>"
                file = file + f"\n<div style='display:inline-block; border-style:solid; border-color:gray;'><p>{souradnice}</p><img src = {server}{image_path} title='{R['day']} {R['type']} {s.tree.value}'></div>"
                images_added = True
    filecontent.value = file
    if images_added:
        solara.Info(
            solara.Markdown(
                f"""
        * File saved and ready to download. 
        * Probes are {probeset}. Image size is {img_size.value}
        * Days are {select_days_multi.value}
        * Pokud by se zobrazovaly obrázky zde, bylo by načítání stránky pomalé kvůli repsonzivnímu designu a množství obrázků. 
          Proto je pohodlnější vytvořit strohý html soubor a zobrazit obrázky v nativní velikosti.  
        """), style={'color': '#2196f3'})
        solara.FileDownload(filecontent.value, filename=f"dynatree_data_acc.html", label="Download")
    else:
        solara.Warning("Vyber aspoň jeden přistroj, aspoň jednu osu a aspoň jeden den v menu nad tímto textem. Stom vyber v boční liště. Ostatní volby z postranní lišty nejsou brány v úvahu.")
    ### The following code works but it is painfuly slow. The server response if fast, probably has something to
    ### due with the number of images which are scaled in browser.
    # for I,R in sets.iterrows():
    #     subsubdf = subdf[(subdf["day"]==R["day"]) & (subdf["type"]==R["type"])]
    #     with solara.Card(title=f"{R['day']} {R['type']}"):
    #         with solara.Row():
    #             for subprobe in probeset:
    #                 with solara.Column():
    #                     sub3df = subsubdf[subsubdf["probe"]==subprobe]
    #                     solara.Markdown(f"**{subprobe}**")
    #                     for i,row in sub3df.iterrows():
    #                         if img_size.value == 'small':
    #                             image_path = "/static/public/cache/FFT_" + row['filename'] + ".png"
    #                         else:
    #                             image_path = "/static/public/fft_images_knocks/FFT_" + row['filename'] + ".png"
    #                         with solara.Row():
    #                             solara.Text(image_path)
    #                             with solara.Column():
    #                                 solara.Text(f"{row['measurement']} @ {row['knock_time']*1.0/100} sec")
    #                                 solara.Text(f"{round(row['freq'])} Hz")



@solara.component
@dynatree.timeit
def Graf():
    if interactive_graph.finished:
        ans = interactive_graph.value
        if ans is None:
            return
    if interactive_graph.not_called:
        return
    with customizable_card(use_overlay.value):
        solara.ProgressLinear(interactive_graph.pending)
        if interactive_graph.finished:
            ans = interactive_graph.value
            current_peaks = rdf.at[ans['index'], "manual_peaks"]
            with solara.Row():
                solara.Button("❌", on_click=lambda: interactive_graph())
                solara.Button("Save & ❌", on_click=lambda: save_click_data(ans['index']))
                solara.Text(ans['text'])
                solara.Text(f"(id {ans['index']})")
            solara.Markdown("**Time domain**")
            solara.FigurePlotly(ans['signal'])
            solara.Markdown(f"""
            **Freq domain and peaks positions**
            
            * Current manual freqs: {[round(i) for i in current_peaks] if current_peaks is not None else "Not defined (yet)."}
            
            * **New manual freqs: {[round(i) for i in manual_freq.value]}**
            """)
            solara.FigurePlotly(ans['fft'], on_click=set_click_data)


def set_click_data(x=None):
    if x['device_state']['shift']:
        manual_freq.value = (x['points']['xs'][0],)
    else:
        manual_freq.value = (*manual_freq.value, x['points']['xs'][0])

@dynatree.timeit
def save_click_data(index):
    global rdf
    rdf.at[index,"manual_peaks"] = manual_freq.value
    df_updated.value = random.random()
    interactive_graph()

@dynatree.timeit
def get_knock_data(**kwds):
    """
    Returns signal after the knock for given measurement_type, day, tree, measurement, knock_time,
    delta_time, probe
    
    Return SignalTuk object
    Knock time is given as int number of 1/100 sec.
    """
    mi = dynatree.DynatreeMeasurement(day=kwds['day'], tree=kwds['tree'], measurement=kwds['measurement'],
                                     measurement_type=kwds['method'])
    knock_time = kwds['knock_time']*1.0/100
    signal_knock = SignalTuk(mi, start=knock_time - delta_time, end=knock_time + delta_time, probe=kwds['probe'])
    return signal_knock

@task
@dynatree.timeit
def interactive_graph(**kwds):
    if len(kwds)==0:
        manual_freq.value = []
        return None
    dynatree.logger.info(f"interactive graph entered {kwds['day']} {kwds['tree']} {kwds['measurement']} {kwds['method']}")
    signal_knock = get_knock_data(**kwds)
    fig1 = px.line(signal_knock.signal,
                   height=200,
                  )
    fft_data = signal_knock.fft
    fig2 = px.line(fft_data,
                   height=300,
                  )
    manual_peaks = rdf.at[kwds['index'], "manual_peaks"]
    df_updated.value = random.random()
    if manual_peaks is not None:
        for _ in manual_peaks:
            fig2.add_vline(x=_, line_width=3, line_dash="dash", line_color="red")

    for fig in [fig1,fig2]:
        fig.update_layout(
            xaxis_title=None,  # Skrývá popisek osy x
            yaxis_title=None,  # Skrývá popisek osy y
            showlegend=False  # Skrývá legendu
        )
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0)  # l = left, r = right, t = top, b = bottom
        )

    fig2.update_yaxes(type="log")  # Logaritmická osa y
    return {'signal':fig1,'fft':fig2,
            'text':f"{kwds['method']}, {kwds['day']}, {kwds['tree']}, {kwds['measurement']}, {kwds['probe']}, {kwds['knock_time']}s",
            'index':kwds['index']
            }


# Funkce pro stylování - přidání hranice, když se změní hodnota v úrovni 'tree'
def add_horizontal_line(df):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)

    # Projdi všechny řádky a přidej stylování
    for i in range(1, len(df)):
        if df.index[i][0] != df.index[i - 1][0]:  # Pokud se změní 'tree'
            styles.iloc[i, :] = 'border-top: 4px solid red'  # Přidej hranici

    return styles

@solara.component
@dynatree.timeit
def Tabulka():
    solara.Markdown(f"# Data for {s.method.value}, {s.day.value}, {s.tree.value}")
    solara.Details(
        summary="Rozklikni pro popis",
        children = [solara.Markdown(
"""
* V tabulce jsou frekvence v Hz. Knock_time je čas v setinách sekundy od začátku měření. 
* Obarvení je po sloupcích. V každém sloupci jsou nejvyšší hodnoty nejtmavší.
* Pod tabulkou je median a další statistiky.
"""
        )]
    )
    subdf = (rdf
        .pipe(lambda d: d[d["tree"] == s.tree.value])
        .pipe(lambda d: d[d["day"] == s.day.value])
        .pipe(lambda d: d[d["type"] == s.method.value])
        #.pipe(lambda d: d[d["measurement"] == s.measurement.value])
        .drop(["day","tree","type","knock_index","filename"], axis=1)
        .pivot(index=['measurement', 'knock_time'], columns='probe', values='freq')
             )
    solara.display(
        subdf
        .style.format(precision=0).background_gradient(axis=0)
        .apply(add_horizontal_line, axis=None)
        .map(lambda x: 'color: lightgray' if pd.isnull(x) else '')
        .map(lambda x: 'background: transparent' if pd.isnull(x) else '')
    )
    with solara.Card(title="Statistics"):
        solara.display(subdf.describe())

first_portrait = solara.reactive(0)
def prev_ten():
    first_portrait.value = max(0,first_portrait.value - cards_on_page.value)
def next_ten(max):
    first_portrait.value = min(max,first_portrait.value + cards_on_page.value)
def prev_next_buttons(max):
    with solara.Row():
        with solara.Columns(1):
            with solara.Column():
                solara.Button(f"Prev {cards_on_page.value}", on_click=prev_ten)
            with solara.Column():
                solara.Button(f"Next {cards_on_page.value}", on_click=lambda: next_ten(max))

select_probe = solara.reactive("All")
select_axis = solara.reactive("All")
select_probe_multi = solara.reactive(["a03"])
select_axis_multi = solara.reactive([])
use_all_measurements = solara.reactive("False")

@solara.component
@dynatree.timeit
def Seznam():
    solara.Markdown(f"""
# Precomputed graphs for {s.method.value} {s.day.value} {s.tree.value}    
    """)
    with solara.Row():
        solara.ToggleButtonsSingle(value=select_probe, values = ["All","a01","a02","a03","a04"])
        solara.ToggleButtonsSingle(value=select_axis, values = ["All","x","y","z"])
        solara.Switch(label="Show all measurements M01, M02, ...", value=use_all_measurements)
    temp_df = (
        rdf
        .pipe(lambda d: d[d["tree"] == s.tree.value])
        .pipe(lambda d: d[d["day"] == s.day.value])
        .pipe(lambda d: d[d["type"] == s.method.value])
        #.drop(["day","tree","type","measurement","knock_index","filename"], axis=1)
    )
    if not use_all_measurements.value:
        temp_df = temp_df[temp_df['measurement'] == s.measurement.value]
    if select_probe.value != "All":
        temp_df = temp_df[temp_df['probe'].str.contains(select_probe.value)]
    if select_axis.value != "All":
        temp_df = temp_df[temp_df['probe'].str.contains(select_axis.value)]
    pocet = len(temp_df)
    prev_next_buttons(max=pocet)
    if first_portrait.value > pocet - 2:
        first_portrait.value = pocet - 5
    for poradi, row in enumerate(temp_df.iterrows()):
        if poradi < first_portrait.value:
            continue
        if poradi > first_portrait.value+cards_on_page.value-1:
            continue
        ReusableComponent(row, poradi, pocet)
    prev_next_buttons(max=pocet)

test_df = rdf.loc[:,["valid","manual_peaks"]]
testdf = solara.reactive(test_df)

@dynatree.timeit
def change_OK_status(i):
    global rdf
    a = time.time()
    current = rdf.at[i,"valid"]
    rdf.at[i,"valid"] = not current
    b = time.time()
    df_updated.value = random.random()
    c = time.time()
    d = time.time()
    dynatree.logger.info(f"Variable setting: {b-a}, copying data: {c-b} versus {d-c}")

@dynatree.timeit
@lru_cache
def FFT_curve(*args, **kwargs):
    knock_data = get_knock_data(**kwargs).fft
    fft_peak = knock_data.iloc[5:].idxmax()
    fig = px.line(x=knock_data.index, y=knock_data.values.reshape(-1), width=400, height=150)
    fig.add_vline(x=fft_peak, line_width=1, line_dash="dash", line_color="green")
    # Odstranění všech dalších prvků
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),  # Nastavení nulových okrajů
        xaxis=dict(visible=False),  # Skrytí osy X
        yaxis=dict(visible=False, type="log"),  # Skrytí osy Y
        showlegend=False,  # Skrytí legendy
        # plot_bgcolor="white",  # Nastavení bílé barvy pozadí
        plot_bgcolor='rgba(0,0,0,0)',  # Nastavení bílé barvy pozadí
    )
    vert_lines = kwargs['vert_lines']
    dynatree.logger.info(f"Vertical lineas are {vert_lines}")
    # # Přidání svislých červených čar
    if vert_lines is not None:
        for _ in vert_lines:
            fig.add_vline(x=_, line_width=1, line_dash="dash", line_color="red")

    return fig


@solara.component
@dynatree.timeit
def ReusableComponent(row, poradi, pocet):
    i, row = row
    is_valid = rdf.at[i, "valid"]
    if is_valid:
        style = {}
        bgstyle = {}
    else:
        style = {'border-right':'solid', 'border-color':'red', 'background-color':'#F0F0F0'}
        bgstyle = {'background-color':'#F0F0F0'}
    image_path = "./static/public/cache/" + row['filename'] + ".png"
    image_path_FFT = "./static/public/cache/FFT_" + row['filename'] + ".png"
    image_path_FFT_large = "./static/public/fft_images_knocks/FFT_" + row['filename'] + ".png"
    fake_variable = df_updated.value
    with solara.Card(style=style):
        if is_valid:
            solara.Text("✅")
        else:
            solara.Text("👎")
        with solara.Row(style=bgstyle):
            with solara.Column(style=bgstyle, gap="0px", classes=['zero-margin']):
                solara.Markdown(
f"""**{poradi + 1}/{pocet}** (id {i})

{row['measurement']} {row['probe']}

@ {row['knock_time']*1.0/100} sec

max at {round(row['freq'])} Hz
""")
                # solara.Text(f"{row['measurement']} {row['probe']}")
                # solara.Text(f"@ {row['knock_time']*1.0/100} sec")
                # solara.Text(f"max at {round(row['freq'])} Hz")
            coordinates = dict(method = s.method.value, day = s.day.value, tree = s.tree.value,
                measurement = row['measurement'], probe = row['probe'],
                knock_time = row['knock_time'], index = i, vert_lines = rdf.at[i,"manual_peaks"])
            if img_from_live_data.value:
                fig = FFT_curve(**coordinates)
                config = {"displayModeBar": False}
                fig.show(config=config)
            elif use_large_fft.value:
                solara.Image(image_path_FFT_large)
            else:
                solara.Image(image_path)
                solara.Image(image_path_FFT)
        man_p = rdf.at[i, "manual_peaks"]
        if man_p is not None:
            solara.Text(f"Manual peaks {[round(_) for _ in man_p]} ", style={'color':'red'})
        with solara.CardActions():
            s.measurement.value = row['measurement']
            with solara.Columns(1):
                with solara.Column():
                    solara.Button("Zadat peaky", text=True, color="primary", on_click=lambda:
                        interactive_graph(**coordinates),
                        outlined=True
                                  )
                with solara.Column():
                    solara.Button("Změnit status", text=True, color="primary", on_click=lambda:
                    change_OK_status(i), outlined=True)

            # solara.Button("Action 2", text=True)

@solara.component
@dynatree.timeit
def Signal():
    solara.Markdown(
"""
# Jedno měření, živě generovaná data

Akcelerometry a02_z a a02_x pro stanovení časů ťuknutí pro dvě sběrnice
""")

    summary_text = "Detailnější popis (klikni pro rozbalení)"
    additional_content = [solara.Markdown(
        """
        * Na obrázku je celý časový průbeh experimentu. Dva kanály a na nich pokusy o nalezení peaků pomocí find_peaks
        a nastavení thresholdů, prominencí a vzdáleností. 
    
        * Do tabulky se pro zpracování ukládají časy ťuků zaokrouhlené na setiny.  Takto je možné přidat později i 
          zatím nerozpoznané ťuky.
    
        * Je možné si zobrazit FFT transformaci, ale dlouho to trvá a lepší je pouužít předpočítané grafy na 
          vedlejší záložce.
        """)
    ]

    solara.Details(
        summary=summary_text,
        children=additional_content,
        expand=False
    )
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
@dynatree.timeit
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

    solara.Button("Plot or Replot", on_click=lambda : plot_all(m, [peak_timesA, peak_timesB]))

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
        for i,knock_time in enumerate(p_times):
            plot_all.progress = (i) * 50.0/n +number*50
            dynatree.logger.warning(f"progress is {(i) * 50.0/n + number*50}")

            all_knocks = [SignalTuk(m, start=knock_time-0.4, end=knock_time+0.4, probe=probe) for probe in probes]
            signals = pd.concat([sg.signal for sg in all_knocks], axis=1)
            ax = signals.plot(subplots=True, sharex=True, figsize=figsize )
            [_.grid() for _ in ax]
            fig1 = plt.gcf()
            fig1.suptitle(f"{m.day} {m.tree} {m.measurement} {m.measurement_type}, ťuk v čase {knock_time}s",
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
            fig2.suptitle(f"{m.day} {m.tree} {m.measurement} {m.measurement_type}, ťuk v čase {knock_time}s",
                        fontsize = 8)
            plt.tight_layout()

            answer[(knock_time,number)] = [fig1, fig2]
    return answer


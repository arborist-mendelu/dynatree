#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 14:00:04 2024

@author: marik
"""

from static_pull import get_all_measurements, available_measurements
import static_pull
import lib_dynatree
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import solara.express as px
import solara.lab
from solara.lab import task
import solara
import time
DATA_PATH = "../data"

tightcols = {'gap': "0px"}
regression_settings = {'color': 'gray', 'alpha': 0.5}
kwds = {"template": "plotly_white", "height": 600}

title = "DYNATREE: vizualizace dat, se kterými se pracuje"


methods = ['normal', 'den', 'noc', 'afterro', 'mraz']
method = solara.reactive('normal')

df = get_all_measurements(method=method.value)
days = df["date"].drop_duplicates().values
trees = df["tree"].drop_duplicates().values
measurements = df["measurement"].drop_duplicates().values


def get_measuerements_list(x='all'):
    global df, days, trees, measurements
    df = get_all_measurements(method='all', type=x)
    days = df["date"].drop_duplicates().values
    trees = df["tree"].drop_duplicates().values
    measurements = df["measurement"].drop_duplicates().values

day = solara.reactive(days[0])
tree = solara.reactive(trees[0])
measurement = solara.reactive(measurements[0])

data_object = lib_dynatree.DynatreeMeasurement(
    day.value, 
    tree.value, 
    measurement.value,
    measurement_type=method.value,
    )

dependent_pull = solara.reactive(["Force(100)"])
dependent_pt34 = solara.reactive(["Pt3"])
dependent_extra = solara.reactive(["Force(100)"])


def resetuj_a_nakresli(x=None):
    pass

def nakresli(x=None):
    pass

def investigate(df, var):
    solara.ToggleButtonsMultiple(value=var, values=list(df.columns))
    px.scatter(df, y=var.value, **kwds)
    df["Time"] = df.index
    solara.DataFrame(df)
    
@solara.component
def Page():
    data_object = lib_dynatree.DynatreeMeasurement(
        day.value, 
        tree.value, 
        measurement.value,
        measurement_type=method.value,
        )
    solara.Title(title)
    solara.Style(".widget-image{width:100%;} .v-btn-toggle{display:inline;}  .v-btn {display:inline; text-transform: none;} .vuetify-styles .v-btn-toggle {display:inline;} .v-btn__content { text-transform: none;}")
    with solara.Sidebar():
        # solara.Markdown("Výběr proměnných pro záložku \"Volba proměnných a regrese\".")
        Selection()
    # solara.Markdown("# Under construction")
    with solara.lab.Tabs():
        with solara.lab.Tab("Tahovky"):
            with solara.Card():
                try:
                    df = data_object.data_pulling
                    investigate(df, dependent_pull)
                # try:
                    # Graphs()
                except:
                    pass
        with solara.lab.Tab("Optika Pt3 a Pt4"):
            with solara.Card():
                try:
                    if data_object.is_optics_available:
                        df2 = data_object.data_optics_pt34
                        df2 = df2-df2.iloc[0,:]
                        df2.columns = [i[0] for i in df2.columns]
                        df2 = df2[[i for i in df2.columns if "Time" not in i]]
                        investigate(df2, dependent_pt34)
                        pass
                    else:
                        solara.Markdown("## Optika pro toto měření není nebo není zpracovaná.")
                # try:
                    # Detail()
                except:
                    pass
        with solara.lab.Tab("Tahovky interpolovane na optiku"):
            with solara.Card():
                try:
                    if data_object.is_optics_available:
                        df3 = data_object.data_optics_extra
                        df3 = df3-df3.iloc[0,:]
                        df3.columns = [i[0] for i in df3.columns]
                        df3 = df3[[i for i in df3.columns if "fixed" not in i]]
                        investigate(df3, dependent_extra)
                        pass
                    else:
                        solara.Markdown("## Optika pro toto měření není nebo není zpracovaná.")
                # try:
                    # Detail()
                except:
                    pass
                # try:
                    # Statistics()
                # except:
                    # pass

        # with solara.lab.Tab("Návod a komentáře"):
        #     with solara.Card():
        #         Help()

    # MainPage()

# def clear():
#     # https://stackoverflow.com/questions/37653784/how-do-i-use-cache-clear-on-python-functools-lru-cache
#     static_pull.nakresli.__dict__["__wrapped__"].cache_clear()
#     static_pull.process_data.__dict__["__wrapped__"].cache_clear()

def Selection():
    with solara.Card():
        with solara.Column():
            solara.ToggleButtonsSingle(value=method, values=list(methods),
                                       on_value=get_measuerements_list)
            solara.ToggleButtonsSingle(value=day, values=list(days),
                                       on_value=resetuj_a_nakresli)
            solara.ToggleButtonsSingle(value=tree, values=list(trees),
                                       on_value=resetuj_a_nakresli)
            solara.ToggleButtonsSingle(value=measurement,
                                       values=available_measurements(
                                           df, day.value, tree.value, method.value),
                                       on_value=nakresli
                                       )
        data_object = lib_dynatree.DynatreeMeasurement(
            day.value, tree.value, measurement.value,measurement_type=method.value)
        solara.Markdown(
            f"**Selected**: {day.value}, {tree.value}, {measurement.value}")
        if data_object.is_optics_available:
            solara.Markdown("✅ Optics is available for this measurement.")
        else:
            solara.Markdown(
                "❎ Optics is **not** available for this measurement.")

def Statistics():
    data_object = get_data_object()
    for df in [data_object.data_optics_extra, data_object.data_pulling]:
        df = df[[i for i in df.columns if "fixed" not in i[0]]]
        nans = pd.DataFrame(df.isna().sum())
        nans.loc[:,"name"] = df.columns
        nans.columns = ["#nan", "name"]
        nans = nans[["name","#nan"]]
        solara.Markdown(f"Shape: {df.shape}")
        solara.DataFrame(nans)


def Graphs():
    solara.ProgressLinear(nakresli.pending)
    if measurement.value not in available_measurements(df, day.value, tree.value, method.value):
        solara.Error(f"""
                     Measurement {measurement.value} not available for tree {tree.value}
                     day {day.value} measurment type {method.value}.
                     """)
        return

    if nakresli.not_called:
        solara.Info(solara.Markdown("""
                        
* Vyber měření a stiskni tlačítko \"Run calculation\". 
* Ovládací prvky jsou v sidebaru. Pokud není otevřený, otevři kliknutím na tři čárky nalevo v modrém pásu.
* Při změně vstupů se většinou obrázek aktualizuje, ale ne vždy. Pokud nadpis na obrázku nesouhlasí s vybranými hodnotami, spusť výpočet tlačítkem \"Run calculation\".

                                    """))
        solara.Warning(
            "Pokud pracuješ v prostředí JupyterHub, asi bude lepší aplikaci maximalizovat. Tlačítko je v modrém pásu úplně napravo.")
    elif not nakresli.finished:
        with solara.Row():
            solara.Markdown("""
            * Pracuji jako ďábel. Může to ale nějakou dobu trvat. 
            * Pokud to trvá déle jak 10 sekund, vzdej to, zapiš si měření a zkusíme zjistit proč. Zatím si prohlížej jiná měření.
            """)
            solara.SpinnerSolara(size="100px")
    else:
        solara.Markdown("Na obrázku je průběh experimentu (časový průběh síly) a detaily pro rozmezí 30%-90% maxima síly. \n\n V detailech je časový průběh síly, časový průběh na inklinometrech a grafy inklinometry versus síla nebo moment. Jestli moment z Rope nebo z tabulky PT viz karta Návod a komentáře.")
        f = nakresli.value
        with solara.ColumnsResponsive(6):
            for _ in f:
                solara.FigureMatplotlib(_)
        plt.close('all')
        solara.Markdown("""
        * Data jsou ze souboru z tahovek. Sample rate cca 0.1 sec. Obrázky jsou jenom pro orientaci a pro kontrolu ořezu dat. Lepší detaily se dají zobrazit na vedlejší kartě s volbou proměnných a regresí.
        * Pokud nevyšla detekce části našeho zájmu, zadej ručně meze, ve kterých hledat. Jsou v souboru `csv/intervals_split_M01.csv` (podadresář souboru se skripty). Potom nahrát na github a zpropagovat do všech zrcadel.
        """
                        )
        # data['dataframe']["Time"] = data['dataframe'].index
        # solara.DataFrame(data['dataframe'], items_per_page=20)
        # cols = data['dataframe'].columns

msg = """
### Něco se nepovedlo. 
                     
* Možná není vybráno nic pro svislou osu. 
* Možná je vybrána stejná veličina pro vodorovnou a svislou osu. 
* Nebo je nějaký jiný problém. Možná mrkni nejprve na záložku Grafy."""

def Detail():
    if nakresli.not_called:
        solara.Info(
            "Nejdřív nakresli graf v první záložce. Klikni na Run calculation v sidebaru.")
        return
    if not nakresli.finished:
        with solara.Row():
            solara.Text("Pracuji jako ďábel. Může to ale nějakou dobu trvat.")
            solara.SpinnerSolara(size="100px")
            return
    # with solara.Card():
    solara.Markdown("## Increasing part of the time-force diagram")
    solara.Markdown("""
            Pro výběr proměnných na vodorovnou a svislou osu otevři menu v sidebaru (tři čárky v horním panelu). Po výběru můžeš sidebar zavřít. Přednastavený je moment vypočítaný z pevného naměřeného úhlu lana na vodorovné ose a oba inlinometry na svislé ose.
            """)

    with solara.Sidebar():
        cols = ['Time', 'Pt3', 'Pt4', 'Force(100)', 'Elasto(90)', 'Elasto-strain',
                # 'Inclino(80)X', 'Inclino(80)Y', 'Inclino(81)X', 'Inclino(81)Y',
                'RopeAngle(100)',
                'blue', 'yellow', 'blueX', 'blueY', 'yellowX', 'yellowY',
                'blue_Maj', 'blue_Min', 'yellow_Maj', 'yellow_Min',
                'F_horizontal_Rope', 'F_vertical_Rope',
                'M_Rope', 'M_Pt_Rope', 'M_Elasto_Rope',
                'F_horizontal_Measure', 'F_vertical_Measure',
                'M_Measure', 'M_Pt_Measure', 'M_Elasto_Measure',]
        with solara.Card():
            solara.Markdown("### Horizontal axis \n\n Choose one variable.")
            solara.ToggleButtonsSingle(values=cols, value=xdata, dense=True)
        with solara.Card():
            solara.Markdown(
                "### Vertical axis \n\n Choose one or more variables. You cannot choose the same variable which has been used for horizontal axis.")
            solara.ToggleButtonsMultiple(
                values=cols[1:], value=ydata, dense=True)
        with solara.Card():
            with solara.VBox():
                with solara.Tooltip("Choose one variable for second vertical axis, shown on the right. (Only limited support in interactive plots. In interactive plots we plot rescaled data. The scale factor is determined from maxima.) You cannot choose the variable used for horizontal axis."):
                    with solara.VBox():
                        solara.Markdown("### Second vertical axis")
                        solara.Text("(hover here for description)")

                solara.ToggleButtonsSingle(
                    values=[None]+cols[1:], value=ydata2, dense=True)
    with solara.Row():
        if measurement.value == "M01":
            with solara.Card():
                with solara.Column(**tightcols):
                    solara.Markdown("Pull No. of M01:")
                    data_object = static_pull.DynatreeStaticMeasurment(
                        day=day.value, tree=tree.value,
                        measurement=measurement.value, measurement_type=method.value,
                        optics=False)
                    print(f"Creating data object {data_object} with pullings {data_object.pullings}")
                    pulls = list(range(len(data_object.pullings)))
                    solara.ToggleButtonsSingle(values=pulls, value=pull)
                pull_value = pull.value
        else:
            pull_value = 0

        with solara.Card():
            with solara.Column(**tightcols):
                solara.Markdown("Bounds to cut out boundaries in % of Fmax")
                solara.ToggleButtonsSingle(
                    values=data_possible_restrictions, value=restrict_data)
        with solara.Card():
            with solara.Column(**tightcols):
                with solara.Tooltip("Umožní zobrazit graf pomocí knihovny Plotly. Bude možné zoomovat, odečítat hodnoty, klikáním na legendu skrývat a odkrývat proměnné apod. Nebudou zobrazeny regresní prímky."):
                    solara.Switch(label="Interactive graph",
                                  value=interactive_graph)
                with solara.Tooltip("Umožní zobrazit grafy veličin pro celý časový průběh."):
                    solara.Switch(
                        label="Ignore time restriction", value=all_data)
        return
        if restrict_data.value != data_possible_restrictions[0]:
            up = .90
            if restrict_data.value == data_possible_restrictions[1]:
                lb = .10
            else:
                lb = .30
            lower, upper = static_pull.get_interval_of_interest(
                subdf, maximal_fraction=up, minimal_fraction=lb)
            subdf = subdf.loc[lower:upper, :]
        if all_data.value:
            subdf = data['dataframe']

    try:
        # find regresions
        if xdata.value != "Time":
            # subsubdf = subdf.loc[:,[xdata.value]+[i for i in ydata.value+[ydata2.value] if i!=xdata.value]]
            ydata.value = [i for i in ydata.value if i != xdata.value]
            if xdata.value == ydata2.value:
                ydata2.value = None
            if ydata2.value is None:
                target = ydata.value
            else:
                target = ydata.value + [ydata2.value]
            reg_df = static_pull.get_regressions(
                subdf,
                [[xdata.value] + target]
            )
            solara.DataFrame(reg_df.iloc[:, :5])
    except:
        solara.Error(
            "Něco se pokazilo při hledání regresí. Nahlaš prosím problém. Pro další práci vyber jiné veličiny.")

    title = f"{day.value} {tree.value} {measurement.value} {method.value} Pull {pull_value}"
    if interactive_graph.value:
        kwds = {"template": "plotly_white", "height": 600, "title": title}
        # kwds = {"height":600, "title":title}
        try:
            if ydata2.value != None:  # Try to add rescaled column
                maximum_target = np.nanmax(
                    subdf[fix_input(ydata.value)].values)
                maximum_ori = np.nanmax(subdf[ydata2.value].values)
                subdf.loc[:, f"{ydata2.value}_rescaled"] = subdf.loc[:,
                                                                     ydata2.value]/np.abs(maximum_ori/maximum_target)
                extradata = [f"{ydata2.value}_rescaled"]
            else:
                extradata = []
            cols_to_draw = fix_input(ydata.value + extradata)
            if xdata.value == "Time":
                px.scatter(subdf, y=cols_to_draw, **kwds)
            else:
                px.scatter(subdf, x=xdata.value, y=cols_to_draw, **kwds)
        except:
            solara.Error(solara.Markdown("""### Image failed. 
                         
* Something is wrong. Switch to noninteractive plot or change variables setting. 
* This error appears especially if you try to plot both forces and inclinometers on vertical axis.
"""
                                         ))
    else:
        fig, ax = plt.subplots()
        if xdata.value == "Time":
            subdf["Time"] = subdf.index
        try:
            subdf.plot(x=xdata.value, y=ydata.value, style='.', ax=ax)
            if xdata.value != "Time":
                t = np.linspace(*ax.get_xlim(), 5)
                for y in ydata.value:
                    if y not in reg_df["Dependent"]:
                        continue
                    d = reg_df[reg_df["Dependent"] ==
                               y].loc[:, ["Slope", "Intercept"]]
                    ax.plot(t, t*d.iat[0, 0]+d.iat[0, 1],
                            **regression_settings)
        except:
            return
            pass
        if ydata2.value != None:
            ax2 = ax.twinx()
            # see https://stackoverflow.com/questions/24280180/matplotlib-colororder-and-twinx
            ax2._get_lines = ax._get_lines
            subdf.plot(x=xdata.value, y=ydata2.value, style='.', ax=ax2)
            # d = reg_df[reg_df["Dependent"]==ydata2.value].loc[:,["Slope","Intercept"]]
            # ax2.plot(t,t*d.iat[0,0]+d.iat[0,1], **regression_settings)
            # ask matplotlib for the plotted objects and their labels
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.get_legend().remove()
            ax2.legend(lines + lines2, labels + labels2)
        ax.grid()
        ax.set(title=title)
        solara.FigureMatplotlib(fig)


def Help():
    solara.Markdown(
        """
## Návod

### Práce

* **Pokud máš otevřeno v zápisníku Jupyteru, bude vhodnější maximalizovat aplikaci. V modrém pásu napravo je ikonka na maximalizaci uvnitř okna prohlížeče.**
* Vyber datum, strom a měření. Pokud se obrázek neaktualizuje automaticky, klikni na tlačítko pro spuštění výpočtu. Výpočet se spustí kliknutím tlačítka nebo změnou volby měření. Pokud se mění strom nebo den a měření zůstává M01, je potřeba stisknout tlačítko.
* Zobrazí se průběh experimentu, náběh (resp. tři náběhy) síly do maxima a zvýrazněná část pro analýzu. Ta je 30-90 procent, ale dá se nastavit i 10-90 procent nebo 0-100 procent.
* Poté máš možnost si detailněji vybrat, co má být v dalším grafu na vodorovné a svislé ose. Tlačítka pro výběr se objeví v bočním panelu, aby se dala skrývat a nezavazela. Počítá se regrese mezi veličinou na vodorovné ose a každou z veličin na ose svislé. Regrese nejsou dostupné, pokud je vodorovně čas (nedávalo by smysl) a pokus je na vodorovné a svislé ose stejná veličina (taky by nedávalo smysl).

### Popis

* Inlinometr blue je 80, yelllow je 81. Výchylky v jednotlivých osách jsou blueX a blueY resp. blue_Maj a blue_Min. Celková výchylka je blue. Podobně  druhý inklinometr.
* F se rozkládá na vodorovnou a svislou složku.Vodorovná se používá k výpočtu momentu v bodě úvazu (M), v bodě Pt3 (M_Pt) a v místě s extenzometrem (M_Elasto). K tomu je potřeba znát odchylku lana od vodorovné polohy. Toto je možné 
    1. zjistit ze siloměru v pull TXT souborech jako Ropeangle(100) 
    2. anebo použít fixní hodnotu z geometrie a měření délek
    3. anebo použít fixní hodnotu naměřenou na začátku experimentu.
* Druhé dvě varianty jsou spolehlivější, **Rope(100) má někdy celkem rozeskákané hodnoty.**  Vypočítané veličiny mají na konci _Rope (varianta 1) nebo _Measure (varianta 3). Varianta 2 bude stejná jsko 3, jenom se lišit konstantním faktorem. Asi se hodí více data s _Measure na konci.
* Elasto-strain je Elasto(90)/200000.

### Komenáře

* V diagramech síla nebo moment versus inklinometry není moc změna trendu mezi první polovinou diagramu a celkem. Takže je asi jedno jestli bereme pro sílu rozmezí 10-90 procent Fmax nebo 10-40 procent.
* Veličina Rope(100) ze siloměru má dost rozeskákané hodnoty a to zašpiní cokoliv, co se pomocí toho počítá. Asi nebrat. To jsou veličiny, které mají na konci text "_Rope". Místo nich použít ty, co mají na konci "_Measure"
* Graf moment versus inlinometry má někdy na začátku trochu neplechu. Možná mají velký vliv nevynulované hodnoty 
  inklinometrů, protože se přidávají k malým náklonům a hodně zkreslují. Zvážit posunutí rozmezí na vyšší hodnotu než 10 procent Fmax.

### Data

Je rozdíl mezi daty ze statiky a pull-release.
Data pro M01 jsou přímo z TXT souborů produkovaných přístrojem. Data pro další 
měření (M02 a výše) byla zpracována: 
    
* počátek se sesynchronizoval s optikou, 
* data se interpolovala na stejné časy jako v optice (tedy je více dat) 
* a někdy se ručně opravilo nevynulování nebo nedokonalé vynulování inklinoměru. 

Dá se snadno přepnout na to, aby se všechna data brala z TXT souborů (volba `skip_optics` ve funkci `get_static_pulling_data`), ale přišli bychom o opravy s vynulováním. Resp. bylo by potřeba to zapracovat.

### Poznámky

|Měření   |Poznámka   |
|:--|:--|
|2021-03-22 BK08 M05| Siloměr neměřil. Není síla ani momenty.|
|2022-08-16 BK13 M02| Optika totálně selhala. TODO: brát jako statiku, viz níže.|
|2022-08-16 BK16 M01| Po zatáhnutí zůstávala velká deformace. Ale zpracování OK.|
|2022-04-05 BK21 M05| Vůbec není v optice. Zatím vyhozeno. TODO: brát jako statiku, viz níže.|

Pokud chceš dynamické měření brát jako statiku, přepni přepínač "Ignore prepocessed files for M2 and higher" (vedle sezamu dostupných měření.)

### Historie

* 2024-08-??: první verze
* 2024-08-23: je možné volit ořez dat v procentech Fmax mezi 0-100%, 10%-90% a 30%-90%, zobrazuje se regresní přímka. TODO: najít v datech, kde se to nejvíce liši a nechat info zde.
* 2024-08-25: zobrazují se i data, ke kterým není nebo zatím není optika, vylepšení ovládání, většinou se výpočet spouští automaticky při změně parametrů
* 2024-08-26: bereme do úvahy i den/noc/afterro/mraz
"""
    )

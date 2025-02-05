#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:41:25 2024

@author: marik
"""
import time
from datetime import timedelta

start_imports = time.time()

import solara
from solara.lab import task
mezicas = time.time() - start_imports
import dynatree.solara.tahovky
import dynatree.solara.force_elasto_inclino
# import dynatree.solara.FFT
import dynatree.solara.welch_ACC
import dynatree.solara.FFT_tukey
import dynatree.solara.download
import dynatree.solara.damping
import dynatree.solara.pulling_tests
import dynatree.solara.tuk_ACC
import dynatree.solara.soil_and_trans
import krkoskova.krkoskova_app
import pandas as pd
from datetime import datetime
import psutil
import os
import toml
mezicas2 = time.time() - start_imports
import dynatree.solara.vizualizace
import_finish = time.time()
print(f"Imports finished in {import_finish-start_imports} sec, mezicas {mezicas}, {mezicas2}")
from dynatree.dynatree import logger
import logging
logger.setLevel(logging.INFO)


with open('solara_texts.toml', 'r') as f:
    config = toml.load(f)

@task
def monitoring():
    def get_metrics():
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = psutil.virtual_memory()
        return cpu_usage, memory_info
    cpu, memory = get_metrics()
    timestamp = time.time()
    dt_object = datetime.fromtimestamp(timestamp)    
    date = f"{dt_object.strftime('%Y-%m-%d %H:%M:%S')}"
    return [cpu, memory, date]
    
@solara.component
def Page():
    logger.info("Page in solara_app.py started")
    with solara.Sidebar():
        solara.Success("Přístup povolen. Vítejte ve zpracování dat projektu Dynatree.")

    with solara.AppBar():
        with solara.Tooltip("Logout"):
            solara.Button(icon_name="mdi-logout",
                          icon=True,
                          attributes={"href": f"/logout"},
                          )

    solara.Title("DYNATREE")
    solara.Markdown(
    """
    Vyber si v menu v barevném panelu nahoře, co chceš dělat.
    """
    )
    with solara.lab.Tabs(vertical=True, background_color=None, dark=False):
        with solara.lab.Tab("Obecné info"):
            solara.Markdown(config['texts']['general_info'])
        # with solara.lab.Tab("Vizualizace"):
        #     solara.Markdown(config['texts']['vizualizace'])
        # with solara.lab.Tab("Tahovky"):
        #     solara.Markdown(config['texts']['tahovky'])
        # with solara.lab.Tab("Synchronizace"):
        #     solara.Markdown(config['texts']['synchronizace'])
        # with solara.lab.Tab("FFT"):
        #     solara.Markdown(config['texts']['FFT'])
        # with solara.lab.Tab("Downloads"):
        #     solara.Markdown(
        #     """
        #     ## Downloads
        #
        #     * Data ke stažení, aktualizují se přímo na serveru po dokončený výpočtů, měla by být vždy ta nejaktuálnější.
        #     """
        #       )

        with solara.lab.Tab("Notes from measurements"):
            solara.Markdown(
            """
            ## Notes from measurements
            
            Notes are extracted from the file `Popis_Babice_VSE.xlsx`.
            The measurements with no notes are dropped. 
            The table is created automatically by snakemake file. The table does not contain
            data annotated by another label than "Measurement:".
            """
              )

            df = pd.read_csv("csv_output/measurement_notes.csv", index_col=0).dropna(subset=["remark1","remark2"], how='all')
            df = df.set_index(["day","tree","measurement"])
#             solara.display(GT(df.reset_index(),
#                               groupname_col="day",
#                               rowname_col="tree",
#                               ).tab_header(
#     title="Large Landmasses of the World",
#     subtitle="The top ten largest are presented"
# ))
            solara.display(df)
        with solara.lab.Tab("Monitoring serveru"):
            with solara.Card():
                if monitoring.not_called:
                    monitoring()
                solara.ProgressLinear(monitoring.pending)
                if monitoring.finished:
                    cpu, memory, date = monitoring.value
                    with solara.Info():
                        with solara.Column():
                            solara.Text(f"CPU Usage: {cpu}%")
                            solara.ProgressLinear(value=cpu, color='red')
                            solara.Text(f"Memory Usage: {memory.percent}%")
                            solara.ProgressLinear(value=memory.percent, color='red')
                            solara.Text(f"Memory Total: {memory.total/1024**3:.2f}GB")
                            solara.Text(f"Cores: {psutil.cpu_count()}")
                            solara.Text(f"Datum a čas: {date}")
                            solara.Text(f"Čas od spuštění/restartu aplikace: {timedelta(seconds = round(time.time()-start_imports))}")
                            try:
                                solara.Text(f"SERVER_NAME {os.environ['SERVER_NAME']}")
                            except:
                                pass
                            try:
                                solara.Text(f"SERVER_PORT {os.environ['SERVER_PORT']}")
                            except:
                                pass
                    solara.Button("Refresh", on_click=monitoring)
    logger.info("Page in solara_app.py finished")


routes = [
    solara.Route(path="/", component=Page, label="home"),
    solara.Route(path="vizualizace", component=dynatree.solara.vizualizace.Page, label="vizualizace"),
    solara.Route(path="tahovky", component=dynatree.solara.tahovky.Page, label="tahovky"),
    solara.Route(path="synchronizace", component=dynatree.solara.force_elasto_inclino.Page, label="synchronizace"),
    # solara.Route(path="FFT_old", component=dynatree.solara.FFT.Page, label="FFT1"),
    solara.Route(path="Welch_ACC", component=dynatree.solara.welch_ACC.Page, label="FFT2"),
    solara.Route(path="FFT_Tukey_all", component=dynatree.solara.FFT_tukey.Page, label="FFT3"),
    solara.Route(path="ACC_tuk", component=dynatree.solara.tuk_ACC.Page, label="ACC_TUK"),
    solara.Route(path="Damping", component=dynatree.solara.damping.Page, label="damping"),
    solara.Route(path="SoilTrans", component=dynatree.solara.soil_and_trans.Page, label="damping"),
    solara.Route(path="Downloads", component=dynatree.solara.download.Page, label="DWNL"),
    solara.Route(path="Krkoskova", component=krkoskova.krkoskova_app.Page, label="K"),
    solara.Route(path="EMA", component=dynatree.solara.pulling_tests.Page, label="EMA"),
]

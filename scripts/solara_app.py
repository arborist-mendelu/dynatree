#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:41:25 2024

@author: marik
"""
import time
from datetime import timedelta
import solara_auth

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
    # if not solara_auth.user.value:
    #     solara_auth.user.value = session_storage.get(solara.get_session_id(), None)
    if solara_auth.needs_login(solara_auth.user.value):
        solara_auth.LoginForm()
        return
    logger.info("Page in solara_app.py started")
    with solara.Sidebar():
        solara.Success("Přístup povolen. Vítejte ve zpracování dat projektu Dynatree.")

    with solara.AppBar():
        with solara.Tooltip("Logout"):
            solara.Button(icon_name="mdi-logout",
                          icon=True,
                          on_click=solara_auth.logout
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


@solara.component
def vizualizace_protected():
    if solara_auth.needs_login(solara_auth.user.value):
        solara_auth.LoginForm()
        return
    dynatree.solara.vizualizace.Page()

@solara.component
def tahovky_protected():
    if solara_auth.needs_login(solara_auth.user.value):
        solara_auth.LoginForm()
        return
    dynatree.solara.tahovky.Page()

@solara.component
def synchro_protected():
    if solara_auth.needs_login(solara_auth.user.value):
        solara_auth.LoginForm()
        return
    dynatree.solara.force_elasto_inclino.Page()

@solara.component
def welch_ACC_protected():
    if solara_auth.needs_login(solara_auth.user.value):
        solara_auth.LoginForm()
        return
    dynatree.solara.welch_ACC.Page()

@solara.component
def FFT_tukey_protected():
    if solara_auth.needs_login(solara_auth.user.value):
        solara_auth.LoginForm()
        return
    dynatree.solara.FFT_tukey.Page()

@solara.component
def tuk_ACC_protected():
    if solara_auth.needs_login(solara_auth.user.value):
        solara_auth.LoginForm()
        return
    dynatree.solara.tuk_ACC.Page()

@solara.component
def damping_protected():
    if solara_auth.needs_login(solara_auth.user.value):
        solara_auth.LoginForm()
        return
    dynatree.solara.damping.Page()

@solara.component
def soil_and_trans_protected():
    if solara_auth.needs_login(solara_auth.user.value):
        solara_auth.LoginForm()
        return
    dynatree.solara.soil_and_trans.Page()

@solara.component
def krkoskova_protected():
    if solara_auth.needs_login(solara_auth.user.value):
        solara_auth.LoginForm()
        return
    krkoskova.krkoskova_app.Page()

@solara.component
def pulling_protected():
    if solara_auth.needs_login(solara_auth.user.value):
        solara_auth.LoginForm()
        return
    dynatree.solara.pulling_tests.Page()

@solara.component
def download_protected():
    if solara_auth.needs_login(solara_auth.user.value):
        solara_auth.LoginForm()
        return
    dynatree.solara.download.Page()

routes = [
    solara.Route(path="/", component=Page, label="home"),
    solara.Route(path="vizualizace", component=vizualizace_protected, label="vizualizace"),
    solara.Route(path="tahovky", component=tahovky_protected, label="tahovky"),
    solara.Route(path="synchronizace", component=synchro_protected, label="synchronizace"),
    # solara.Route(path="FFT_old", component=dynatree.solara.FFT.Page, label="FFT1"),
    solara.Route(path="Welch_ACC", component=welch_ACC_protected, label="FFT2"),
    solara.Route(path="FFT_Tukey_all", component=FFT_tukey_protected, label="FFT3"),
    solara.Route(path="ACC_tuk", component=tuk_ACC_protected, label="ACC_TUK"),
    solara.Route(path="Damping", component=damping_protected, label="damping"),
    solara.Route(path="SoilTrans", component=soil_and_trans_protected, label="damping"),
    solara.Route(path="Downloads", component=download_protected, label="DWNL"),
    solara.Route(path="Krkoskova", component=krkoskova_protected, label="K"),
    solara.Route(path="EMA", component=pulling_protected, label="EMA"),
]

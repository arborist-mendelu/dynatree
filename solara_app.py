#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:41:25 2024

@author: marik
"""

import solara
from solara.lab import task
import lib.solara.tahovky
import lib.solara.vizualizace
import lib.solara.force_elasto_inclino
import lib.solara.FFT
import lib.solara.welch_ACC
import lib.solara.FFT_tukey
import lib.solara.download
import krkoskova.krkoskova_app
from passlib.hash import pbkdf2_sha256
import pandas as pd
import time
from datetime import datetime
import os
import psutil

import toml
with open('solara_texts.toml', 'r') as f:
    config = toml.load(f)

def Naloguj_se():
    solara.Title("DYNATREE")
    solara.Warning(solara.Markdown(config['texts']['login_info']))
    with solara.Sidebar():
        Login()

@solara.component
def Ochrana(funkce):
    if user_accepted.value:
        Logout()
        funkce()
    else:
        Naloguj_se()

user_accepted = solara.reactive(False)
password_attempt = solara.reactive("")
valid_hashes = [
 "$pbkdf2-sha256$29000$rbU2prR2jhGidK4VgnAu5Q$e.CvUxgiY3uImVIuUTrKYFWRh/eak5oNVS.WMbBt3mI", 
 "$pbkdf2-sha256$29000$/F9LSYkx5rx3TmlNiZGSUg$YO/PMhUB9imJjjqoZC48OGLn3UOYq8GmnxhMDdEi9eo",
 "$pbkdf2-sha256$29000$vTem1BojhJCS0vo/B.D8fw$F/XHKhni22p9.kfiRB/c9WqgLMg.NkhgLBr/eTnPmsU",
 "$pbkdf2-sha256$29000$LaXUOmfMGQNAaK31PmesdQ$/CKKV.V6J3SkaUaKI.UEIOub2rYdyI/tcznMzMWF27s",
   ]

@solara.component
def Login():
    solara.Text("Zadej heslo")
    solara.InputText("Heslo", value=password_attempt, password=True)
    test_login = [
        pbkdf2_sha256.verify(password_attempt.value, i) for i in valid_hashes
        ]
    try:
        if os.environ['SERVER_NAME'] == "localhost":
            test_login = [True]  # auto login for everybody on local host
    except:
        pass
    if True in test_login:
        user_accepted.value = True
        solara.Success(solara.Markdown("Acess granted"))
        password_attempt.value=""
    else:
        if password_attempt.value != "":
            solara.Error(solara.Markdown(
                f"""
                * Enter valid password.
                * Login failed.
                * {time.ctime()}
                """))
            # solara.Info(solara.Markdown(
            #     f"""
            #     **Your input was not recognised as a valid password.**
            #     """
            #     ))

def Logout():
    with solara.AppBar():
        if user_accepted.value:
            try:
                solara.Text(f"Solara on {os.environ['SERVER_NAME']}")
            except:
                pass
            with solara.Tooltip("Logout"):
                solara.Button(icon_name="mdi-logout",
                              icon=True, on_click=lambda: user_accepted.set(False))    

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

    try:
        servername = os.environ['SERVER_NAME']
        if servername == "localhost":
            solara.lab.theme.themes.light.primary = "#006000"
        elif servername == "um-bc201.mendelu.cz":
            solara.lab.theme.themes.light.primary = "#000000"
    except:
        pass

    with solara.Sidebar():
        solara.Markdown("**Projekt DYNATREE**")
        if not user_accepted.value:
            Login()
        else:
            solara.Success("Acess granted.")
    # with solara.Column(): 
    if True:
        solara.Title("DYNATREE")
        solara.Markdown(
        """
        Vyber si v menu v barevném panelu nahoře, co chceš dělat. Ve svislém menu níže je stručný popis.
        """
        ) 
        with solara.lab.Tabs(vertical=True, background_color=None, dark=False):
            with solara.lab.Tab("Obecné info"):                    
                solara.Markdown(config['texts']['general_info'])
            with solara.lab.Tab("Vizualizace"):                    
                solara.Markdown(config['texts']['vizualizace'])    
            with solara.lab.Tab("Tahovky"):
                solara.Markdown(config['texts']['tahovky'])
            with solara.lab.Tab("Synchronizace"):
                solara.Markdown(config['texts']['synchronizace'])
            with solara.lab.Tab("FFT"):
                solara.Markdown(config['texts']['FFT'])
            with solara.lab.Tab("Downloads"):
                solara.Markdown(
                """
                ## Downloads
                
                * Data ke stažení, aktualizují se přímo na serveru po dokončený výpočtů, měla by být vždy ta nejaktuálnější. 
                """
                  )

            with solara.lab.Tab("Notes from measurements"):
                solara.Markdown(
                """
                ## Notes from measurements
                
                Notes are extracted from the file `Popis_Babice_VSE_13082024.xlsx`.
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
                        solara.Button("Refresh", on_click=monitoring)
                
routes = [
    solara.Route(path="/", component=Ochrana(Page), label="home"),
    solara.Route(path="vizualizace", component=Ochrana(lib.solara.vizualizace.Page), label="vizualizace"),
    solara.Route(path="tahovky", component=Ochrana(lib.solara.tahovky.Page), label="tahovky"),
    solara.Route(path="synchronizace", component=Ochrana(lib.solara.force_elasto_inclino.Page), label="synchronizace"),
    solara.Route(path="FFT_old", component=Ochrana(lib.solara.FFT.Page), label="FFT1"),
    solara.Route(path="Welch_ACC", component=Ochrana(lib.solara.welch_ACC.Page), label="FFT2"),
    solara.Route(path="FFT_Tukey_all", component=Ochrana(lib.solara.FFT_tukey.Page), label="FFT3"),
    solara.Route(path="Downloads", component=Ochrana(lib.solara.download.Page), label="DWNL"),
    solara.Route(path="Krkoskova", component=Ochrana(krkoskova.krkoskova_app.Page), label="K"),
]

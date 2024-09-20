#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 18:41:25 2024

@author: marik
"""

import solara
import solara_major_minor_momentum
import solara_vizualizace
import solara_force_elasto_inclino
import solara_FFT
import solara_FFT_ACC
import solara_FFT_tukey
from passlib.hash import pbkdf2_sha256
import pandas as pd
import time
import os

def Naloguj_se():
    solara.Title("DYNATREE")
    solara.Warning(solara.Markdown(
        """
        ## Naloguj se v bočním menu 
        
        * Tato část webu není veřejná
        * Heslo je obvyklé. Pokud nevíš, zeptej se někoho kolem sebe.
        * Pokud chceš používat vlastní heslo, napiš Robertovi.
        * Pokud chceš nastavit vlastní **tajné** heslo, vygeneruj hash a pošli Robertovi. 
          K tomu použij následující příkazy.
        
          ~~~
          from passlib.hash import pbkdf2_sha256
          hash = pbkdf2_sha256.hash("moje_super_tajne_heslo_ktere_nikomu_nereknu")
          print(hash)
          ~~~

        
        """))
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

@solara.component
def Page():
    Logout()
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
                    
                solara.Markdown(
                """
                ## Obecné info
                
                * Někdy se po přepnutí aplikací neaktualizuje menu v levém sidebaru. 
                  To je možné opravit kliknutím na jinou položku v typu měření (normal/den/noc/...)
                * Někdy se objeví chybová hláška, ale hned zmizí, to je neškodné. 
                * U aplikací, které se spouští automaticky výběrem dne/stromu/měření může při rychlém klikání beh skončit chybou. Zatím není pořešeno odstřelování neaktuálních procesů.
                """)
            with solara.lab.Tab("Vizualizace"):
                    
                solara.Markdown(
                """
                ## Vizualizace            
                
                * Grafy z dat pro jednotlivá měření.
                * Obsahuje 
                    * optiku Pt3 a Pt4, 
                    * tahovky, 
                    * tahovky dointerpolované na data z optiky.
                * U každého druhu dat si můžeš vybrat veličiny na svislou osu, grafy zoomovat apod.
                * Použití: vykreslení jedné nebo několika položek, kontrola (např. vynulování u přistrojů, které se nulují), hledání peaků, hledání intervalů zájmu (dají se vybrat data nástrojem a souřadnice výběru uložit pro pozdější použití).
                """
                )    
            with solara.lab.Tab("Tahovky"):
                solara.Markdown(
                """
                ## Tahovky
                
                * Obsahuje data z tahových zkoušek ze všech měření. 
                * Je zpracovávána jedna nebo tři napínací fáze.
                * Záložky obsahují základní přehled, možnost zobrazit si detail s vybranými
                   veličinami na vodorovné a svislé ose a jsou vypočteny regresní koeficienty.
                * Použití: vyexportujeme si všechny regrese, ale bude jich hodně. Záložka umožní kouknout se na data, která nás zajímají, protože vyšla například nějak divně. Případně k vizuální kontrole, jestli dostáváme to co chceme. 
                """
                )    
            with solara.lab.Tab("Synchronizace"):
                solara.Markdown(
                """
                ## Synchronizace
                
                * Obsahuje spojení dat z tahovek a optiky
                * Používá se ke kontrole vynulování inklinometrů a ke kontrole synchronizace.
                * Předpočítaná data se ignorují a zohledňují se data z csv souboru v podadreáři csv. Sem je možné ručně připsat potřebné opravy. Poté se data projedou skriptem, který vytvoří data synchroniozvaná s optikou a tato data se potom používají všude jinde.
                * Obrázky se dají vyexportovat skriptem `plot_probes_inclino_force.py` k vizuální kontrole jako pdf nebo png.
                * Použití: Aby se daly společně vyhodnocovat data z optiky a tahovek, je potřeba je mít seskupitelná dohromady.
                """
                )    
            with solara.lab.Tab("FFT"):
                solara.Markdown(
                """
                ## FFT
                
                * Umožní udělat FFT na zvolelných datech. Obsahuje optiku (Pt3, Pt4, konce BL) a tahovky (Elasto).
                * Data se interpolují s krokem 0.01s.
                * Použití: rychlé zobrazení spekter, vizuální porovnání, zjištění důvodů, proč se některá spektra liší (například kvůli krátkému sigálu, narušení oscilací apod.) 
                """
                  )

            with solara.lab.Tab("Notes from measurements"):
                solara.Markdown(
                """
                ## Notes from measurements
                
                Notes are extracted from the file `Popis_Babice_VSE_13082024.xlsx`.
                The measurements with no notes are dropped. 
                The table is created automatically by snakemake file.
                """
                  )
                
                df = pd.read_csv("csv_output/measurement_notes.csv", index_col=0).dropna(subset=["remark1","remark2"], how='all')
                df = df.set_index(["day","tree","measurement"])
                solara.display(df)


routes = [
    solara.Route(path="/", component=Page, label="home"),
    solara.Route(path="vizualizace", component=Ochrana(solara_vizualizace.Page), label="vizualizace"),
    solara.Route(path="tahovky", component=Ochrana(solara_major_minor_momentum.Page), label="tahovky"),
    solara.Route(path="synchronizace", component=Ochrana(solara_force_elasto_inclino.Page), label="synchronizace"),
    solara.Route(path="FFT", component=Ochrana(solara_FFT.Page), label="FFT1"),
    solara.Route(path="FFT_for_ACC", component=Ochrana(solara_FFT_ACC.Page), label="FFT2"),
    solara.Route(path="FFT_Tukey_all", component=Ochrana(solara_FFT_tukey.Page), label="FFT3"),
]

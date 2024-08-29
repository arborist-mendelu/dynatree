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
from passlib.hash import pbkdf2_sha256
import time

def Naloguj_se():
    solara.Title("DYNATREE")
    solara.Warning(solara.Markdown(
        """
        ##Naloguj se. 
        
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
   ]

@solara.component
def Login():
    global failed_attempts
    solara.InputText("Heslo", value=password_attempt, password=True)
    hash_attempt = pbkdf2_sha256.hash(password_attempt.value)
    test_login = [
        pbkdf2_sha256.verify(password_attempt.value, i) for i in valid_hashes
        ]
    if True in test_login:
        user_accepted.value = True
        solara.Success(solara.Markdown("Acess granted"))
        password_attempt.value=""
    else:
        if password_attempt.value != "":
            solara.Error(solara.Markdown(
                f"""
                * Enter valid password.
                * Login on failed.
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
    with solara.Card(title = "Dynatree data"): 
        solara.Title("DYNATREE")
        solara.Markdown(
        """
  
        Vyber si v menu.
    
        ## Vizualizace            
        
        * Grafy z dat pro jednotlivá měření.
        * Obsahuje 
            * optiku Pt3 a Pt4, 
            * tahovky, 
            * tahovky dointerpolované na data z optiky.
        * U každého druhu dat si můžeš vybrat veličiny na svislou osu, grafy zoomovat apod.
        * Použití: vykreslení jedné nebo několika položek, kontrola (např. vynulování u přistrojů, které se nulují), hledání peaků.
    
        ## Tahovky
        
        * Obsahuje data z tahových zkoušek ze všech měření. 
        * Je zpracovávána jedna nebo tři napínací fáze.
        * Záložky obsahují základní přehled, možnost zobrazit si detail s vybranými
           veličinami na vodorovné a svislé ose a jsou vypočteny regresní koeficienty.
        * Použití: vyexportujeme si všechny regrese, ale bude jich hodně. Záložka umožní kouknout se na data, která nás zajímají, protože vyšla například nějak divně. Případně k vizuální kontrole, jestli dostáváme to co chceme. 
        
        # Synchronizace
        
        * Obsahuje spojení dat z tahovek a optiky
        * Používá se ke kontrole vynulování inklinometrů a ke kontrole synchronizace.
        * Předpočítaná data se ignorují a zohledňují se data z csv souboru v podadreáři csv. Sem je možné ručně připsat potřebné opravy. Poté se data projedou skriptem, který vytvoří data synchroniozvaná s optikou a tato data se potom používají všude jinde.
        * Obrázky se dají vyexportovat skriptem `plot_probes_inclino_force.py` k vizuální kontrole jako pdf nebo png.
        * Použití: Aby se daly společně vyhodnocovat data z optiky, je potřeba je mít seskupitelná dohromady.
        
        """
          )

routes = [
    solara.Route(path="/", component=Page, label="home"),
    solara.Route(path="vizualizace", component=Ochrana(solara_vizualizace.Page), label="vizualizace"),
    solara.Route(path="tahovky", component=Ochrana(solara_major_minor_momentum.Page), label="tahovky"),
    solara.Route(path="synchronizace", component=Ochrana(solara_force_elasto_inclino.Page), label="synchronizace"),
]
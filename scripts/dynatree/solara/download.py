#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:24:26 2024

@author: marik
"""

import solara
import yaml
import os
from datetime import datetime

def get_file_modification_date(file_path):
    """
    Vrátí datum a čas poslední modifikace souboru.

    :param file_path: Cesta k souboru
    :return: Datum a čas poslední modifikace jako řetězec
    """
    try:
        timestamp = os.path.getmtime(file_path)
        modification_date = datetime.fromtimestamp(timestamp)
        return modification_date.strftime('%Y-%m-%d %H:%M:%S')
    except FileNotFoundError:
        return "???"
    except Exception as e:
        return "???"

# Funkce pro získání velikosti souboru v MB na 2 desetinná místa
def velikost_souboru_v_mb(cesta_k_souboru):
    try:
        velikost_bajty = os.path.getsize(cesta_k_souboru)  # Získání velikosti souboru v bajtech
        velikost_mb = velikost_bajty / (1024 * 1024)  # Převod na MB
        return f"{velikost_mb:.2f} MB"  # Formátování na 2 desetinná místa
    except:
        return "?? MB"

@solara.component
def Page():
    # with open('downloads.yml', 'r') as file:
    #     data = yaml.load_all(file, yaml.FullLoader)
    solara.Title("DYNATREE: Download site")
        # solara.Style(s.styles_css)
    solara.Markdown("# Downloads")
    
    with solara.Info():
        solara.Markdown(
"""
* Výstupy ze skriptů v projektu. Jsou zde soubory, které vznikají při běhu `snakemake` v adresáři
  `outputs`. 
* Odkazy se sem přidávají ručně, tak všechno nemusí být aktuální. Data se potom aktualizují
  při `snakemake`.
* Odkazy fungují i bez nutnosti zadávat heslo.
""", style={'color':'inherit'})

    # title = {0: "Aktuální soubory", 1: "Asi už nepotřebné soubory"}

    file = open('downloads.yml', 'r')
    data = yaml.load_all(file, yaml.FullLoader)

    for t,doc in enumerate(data):
        # solara.Markdown(f"## {title[t]}")
        with solara.Card():
            for k,v in doc.items():
                if k=="title":
                    solara.Markdown(f"## {v}")
                elif k=="popis":
                    solara.Markdown(v)
                else:
                    print(k)
                    _ = v.split(".")
                    popis = _[0]
                    detail = ". ".join(_[1:])
                    with solara.Column(gap='20px'):
                        lightgray = "#F4F4F4"
                        with solara.Card(style={"background-color": lightgray}):
                            with solara.Row(style={"background-color": lightgray}):
                                if "http" in k:
                                    size = ""
                                    solara.Button(label="Open URL",
                                                  attributes={"href": k, "target": "_blank"},
                                                  color='primary')
                                else:
                                    solara.Button(
                                        label=f"Download ({velikost_souboru_v_mb('../outputs/'+k)})",
                                        attributes={"href": f"./static/public/{k}", "target": "_blank"}, color='primary')
                                    size = f"(*Version {get_file_modification_date('../outputs/'+k)}*)"
                                solara.Markdown(f"""
**{popis}.** {detail} {size}
""")


    
        
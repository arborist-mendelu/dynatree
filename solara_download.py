#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 17:24:26 2024

@author: marik
"""

import solara
import yaml



@solara.component
def Page():
    # with open('downloads.yml', 'r') as file:
    #     data = yaml.load_all(file, yaml.FullLoader)
    solara.Title("DYNATREE: FFT")
    # solara.Style(s.styles_css)
    solara.Markdown("# Downloads")
    
    solara.Info("""Stahuj pomocí pravého tlačítka a "otevřít v novém panelu". Jinak 
                se nepůjde přepnout na jinou záložku a budeš muset dát reload a znovu zadávat
                heslo.                
                """)

    title = {0: "Aktuální soubory", 1: "Asi už nepotřebné soubory"}

    file = open('downloads.yml', 'r')
    data = yaml.load_all(file, yaml.FullLoader)
    # for doc in data:
    #     for k,v in doc.items():
    #         print(k, "->", v)
    #         print(f"**🔗 <a href=/static/public/{k}>{k}</a>**: {v}")
    #     print ("\n")
    # with solara.Column(gap='0px'):
    for t,doc in enumerate(data):
        solara.Markdown(f"## {title[t]}")
        for k,v in doc.items():
            solara.Markdown(f"**🔗 <a href=/static/public/{k}>{k}</a>**: {v}")


        
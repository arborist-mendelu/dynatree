""""
Program načte soubor ../../data/puda/Puda/vlhkosti_babice.ods
Načte data ze všech listů, které začínají číslem. 
"""

# import knihoven
import pandas as pd
import os

# cesta k souboru
cesta = os.path.join("..", "..", "data", "puda", "Puda", "vlhkosti_babice.ods")

# načtení dat
data = pd.read_excel(cesta, sheet_name=None, header=None)

# názvy listů, které začínají číslem
seznam_listu = [list for list in data.keys() if list[0].isdigit()]

# vytvoření prázdného DataFrame
vysledky = pd.DataFrame()

# procházení listů
for den in seznam_listu:
    # načtení dat s vynecháním prvních dvou řádků
    data_list = data[den].iloc[2:,:].copy()
    # přidání sloupce s názvem listu
    data_list["den"] = den
    # přidání dat do výsledného DataFrame
    vysledky = pd.concat([vysledky, data_list])
    
# přejmenování sloupců podle druhého řádku prvního listu
vysledky.columns = [i for i in data[seznam_listu[0]].iloc[1,:]] + ["den"]
# Konverze m¹ m² na m1 a m2 v názvech sloupců
vysledky.columns = [i.replace("¹", "1").replace("²", "2") for i in vysledky.columns]
# Konverze 20210101_mokro na 2021-01-01_mokro ve sloupci den
vysledky["den"] = vysledky["den"].apply(lambda x: f"{x[:4]}-{x[4:6]}-{x[6:]}")
# uložení výsledků
vysledky.to_csv("../../outputs/vlhkosti_babice.csv", index=False)

data = {"sonda": [1,2,3,4,4,5,5,5,6,6,7,7,8,9],
        "strom": [1,4,7,8,9,10,11,12,13,14,16,18,21,24]}

data = pd.DataFrame(data)
data.to_csv("../../outputs/sondy_a_stromy.csv", index=False)




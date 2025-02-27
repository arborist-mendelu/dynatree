import pandas as pd
import numpy as np

# path = "../"
import os
# os.environ["PREFIX_DYNATREE"] = path
# import sys
# sys.path.append(path)

from dynatree import find_measurements

df = find_measurements.get_all_measurements(method='all', type='all')

import requests
from urllib.parse import urlencode

# Načtení tabulky df (předpokládáme, že už existuje)
# df = pd.read_csv("data.csv")  # Pokud je potřeba načíst tabulku

# Adresář pro ukládání obrázků
output_dir = "figs"
os.makedirs(output_dir, exist_ok=True)

# Základní URL
base_url = "https://euler.mendelu.cz/draw_graph_damping/"

# Stažení obrázků
for _, row in df.iterrows():
    params = {
        "method": f"{row['day']}_{row['type']}",
        "tree": row["tree"],
        "measurement": row["measurement"],
        "probe": "Elasto(90)",
        "format": "png",
        "damping_method": "extrema"
    }
    if row["measurement"] == "M01":
        continue
    url = f"{base_url}?{urlencode(params)}"
    filename = f"{params['method']}_{row['tree']}_{row['measurement']}.png"
    filepath = os.path.join(output_dir, filename)

    try:
        response = requests.get(url, timeout=2)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
        print(f"Uloženo: {filename}")
    except requests.RequestException as e:
        print(f"Chyba při stahování {filename}: {e}")
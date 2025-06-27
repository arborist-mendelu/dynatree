import os

import pandas as pd
import requests
from urllib.parse import urlencode
from PIL import Image
from io import BytesIO
from dynatree import find_measurements

# Probe
probes = ["Pt3", "Pt4", "blueMaj", "yellowMaj"]
probe = probes[3]


# Načtení tabulky df
df = find_measurements.get_all_measurements(method='all', type='all')

# Adresář pro ukládání obrázků
output_dir = "figs"
os.makedirs(output_dir, exist_ok=True)

# Základní URL
base_url1 = "https://euler.mendelu.cz/draw_graph_damping/"
base_url2 = "https://euler.mendelu.cz/draw_graph/"


manual_data = pd.read_csv("csv/damping_manual_ends.csv", skipinitialspace=True).iloc[:,:-1].values
manual_data = [tuple(i) for i in manual_data]

# Stažení a spojení obrázků
for _, row in df.iterrows():
    if row["measurement"] == "M01":
        continue
    
    params = {
        "method": f"{row['day']}_{row['type']}",
        "tree": row["tree"],
        "measurement": row["measurement"],
        "probe": probe,
        "format": "png",
        "damping_method": "extrema"
    }

#    if not ( (row['type'], row['day'], row['tree'], row['measurement']) in manual_data ):
#    if not row['day'] == "2025-04-01":
#        continue

    url1 = f"{base_url1}?{urlencode(params)}"
    url2 = f"{base_url2}?{urlencode(params)}"
    
    filename = f"{params['method']}_{row['tree']}_{row['measurement']}_{probe}.png"
    filepath = os.path.join(output_dir, filename)
    
    try:
        response1 = requests.get(url1, timeout=10)
        response1.raise_for_status()
        img1 = Image.open(BytesIO(response1.content))
        
        response2 = requests.get(url2, timeout=10)
        response2.raise_for_status()
        img2 = Image.open(BytesIO(response2.content))
        
        # Spojení obrázků pod sebe
        width = max(img1.width, img2.width)
        height = img1.height + img2.height
        combined_img = Image.new("RGB", (width, height))
        combined_img.paste(img1, (0, 0))
        combined_img.paste(img2, (0, img1.height))
        
        # Uložení výsledného obrázku
        combined_img.save(filepath)
        print(f"Uloženo: {filename}")
    except requests.RequestException as e:
        print(f"Chyba při stahování obrázků: {e}")

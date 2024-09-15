#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 03 2024

Skript hledá acc soubory a prepisuje je do parquet. Nemeni sampling rate.


@author: marik
"""

import glob 
import os
import pandas as pd
import scipy.io
import logging
logger = logging.getLogger("prevod")
logger.setLevel(logging.DEBUG)
from pathlib import Path
from tqdm import tqdm

from scipy.signal import savgol_filter

source_dir = "/mnt/ERC/ERC"

adresar = source_dir+'/Mereni_Babice'

logger.debug("Starting to search directory")

matching_dirs = glob.glob(os.path.join(adresar, '**/ACC'), recursive=True)

def data(text):
    t = text.split("/")
    tree = t[-2]
    date = t[-5].replace("_","-")
    kind = t[-3]
    return [date, kind, tree, text]

upraveny_seznam = [data(i) for i in matching_dirs]
df = pd.DataFrame(upraveny_seznam, columns=["date","kind","tree","directory"])

#%%

pbar = tqdm(total=len(df))
for i,row in df.iterrows():
    files = glob.glob(os.path.join(row['directory'], '*.mat'))
    for soubor in files:
        # print(soubor)
        measurement = soubor.split("/")[-1].split(".")[-2]
        target = f"../data/parquet_acc/{row['kind'].lower()}_{row['date']}_{row['tree']}_{measurement}_5000.parquet"
        file_path = Path(target)
        if file_path.is_file():
            # print(f"Soubor {target} existuje, koncim.")
            continue
        logger.info(f"Loading {soubor}")
        # continue
        mat = scipy.io.loadmat(soubor)
        data = {k:mat[k].T[0] for k in mat.keys() if "Data1_A" in k}
        length = min([len(j) for j in data.values()])
        data = {k:data[k][:length] for k in data.keys()}        
        newdf = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in data.items()]))
        newdf.to_parquet(target)
        # logger.info(f"File {soubor} processed")        
    pbar.update(1)

pbar.close()
logging.info("Finished")
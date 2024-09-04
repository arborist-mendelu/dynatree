#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 03 2024

Skript hledá acc soubory a prepisuje je do parquet. Meni frekvenci z 5000 na 100.


@author: marik
"""

import glob 
import os
import pandas as pd
import scipy.io
import logging
logger = logging.getLogger("prevod")
logger.setLevel(logging.DEBUG)

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

for i,row in df.iterrows():
    files = glob.glob(os.path.join(row['directory'], '*.mat'))
    for soubor in files:
        print(soubor)
        measurement = soubor.split("/")[-1].split(".")[-2]
        mat = scipy.io.loadmat(soubor)
        logger.info(f"File {soubor} loaded")
        data = {k:mat[k].T[0] for k in mat.keys() if "Data1_A" in k}
        smoothed_data = {k: savgol_filter(data[k], window_length=1000, polyorder=2)[::50] for k in data.keys()}  # Změna hodnot window_length a polyorder může ovlivnit úroveň vyhlazení
        newdf = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in smoothed_data.items()]))
        target = f"../data/parquet_acc/{row['kind'].lower()}_{row['date']}_{row['tree']}_{measurement}.parquet"
        newdf.to_parquet(target)
        logger.info(f"File {soubor} processed")        

logging.info("Finished")
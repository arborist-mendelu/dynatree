"""
Ulozi fft obrazky a obrazky kmitu do souborove cache.
"""
import sys
import gc
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import time

import dynatree.dynatree
from config import file
import matplotlib.pyplot as plt
from parallelbar import progress_map
import logging
dynatree.dynatree.logger.setLevel(logging.ERROR)

delta_time = 0.4
df = pd.read_csv(file['outputs/FFT_acc_knock'])
df = df[df["type"]!="testovaci"]

from dynatree.dynatree import DynatreeMeasurement
from dynatree.signal_knock import SignalTuk

cachedir = file['cachedir']
cachedir_large = file['cachedir_large']
# cachedir = "/home/marik/dynatree_outputs/new_cache"
# cachedir_large = "/home/marik/dynatree_outputs/new_fft_images_knocks"

# @dynatree.dynatree.timeit
def save_images(signal_knock, fft_peak, figname):
    if (
            Path(f"{cachedir}/{figname}.png").exists() &
            Path(f"{cachedir}/FFT_{figname}.png").exists() &
            Path(f"{cachedir_large}/FFT_{figname}.png").exists()
    ):
        pass
        # return
        # TODO: ukoncit vypocet drive, pred nactenim acc dat.
    # small time domain
    fig, ax = plt.subplots(figsize=(3,1))
    if "_x_" in figname:
        color="C0"
    elif "_y_" in figname:
        color="C1"
    else:
        color="C2"
    ax.plot(signal_knock.signal, color=color)
    fig.savefig(f"{cachedir}/{figname}.png", transparent=True)
    # print(f"{cachedir}/{figname}.png saved")

    # small FFT
    signal_fft = signal_knock.fft
    fig, ax = plt.subplots(figsize=(3,1))
    if "_x_" in figname:
        color="C0"
    elif "_y_" in figname:
        color="C1"
    else:
        color="C2"
    ax.plot(signal_fft, color=color)
    ax.axvline(x=fft_peak, color='r', linestyle='--')
    ax.grid()
    ax.set(yscale='log')
    fig.savefig(f"{cachedir}/FFT_{figname}.png", transparent=True)
    # print(f"{cachedir}/FFT_{figname}.png saved")

    # large FFT
    fig, ax = plt.subplots(figsize=(9,3))
    if "_x_" in figname:
        color="C0"
    elif "_y_" in figname:
        color="C1"
    else:
        color="C2"
    ax.plot(signal_fft, color=color)
    ax.axvline(x=fft_peak, color='r', linestyle='--')
    ax.grid()
    ax.set(yscale='log')
    fig.savefig(f"{cachedir_large}/FFT_{figname}.png", transparent=True)
    # print(f"{cachedir_large}/FFT_{figname}.png saved")

    plt.close('all')


# @dynatree.dynatree.timeit
def zpracuj_mereni(row):
    columns_to_match = ["type", "tree", "day", "measurement"]
    subdf = df[(df[columns_to_match] == row[columns_to_match]).all(axis=1)]
    m = DynatreeMeasurement(day=row["day"], tree=row["tree"], measurement=row["measurement"],
                            measurement_type=row["type"])
    accdata = m.data_acc5000 # TODO: toto presunout do cyklu a pred to dat continue, pokud soubor existuje.
    for subi,subrow in subdf.iterrows():
        s = SignalTuk(m, subrow["knock_time"] / 100.0 - delta_time,
                      subrow["knock_time"] / 100.0 + delta_time,
                      subrow["probe"], accdata=accdata)
        save_images(s, subrow['freq'], subrow['filename'])

def process_chunk(df):
    ans = progress_map(zpracuj_mereni, [i for _,i in df.iterrows()], disable=True)
    return ans

def process_df(df):
    n = 10  # chunk row size
    list_df = [df[i:i + n] for i in range(0, df.shape[0], n)]
    delka = len(list_df)
    i = 0
    start = time.time()
    ans = {}
    for _,d in enumerate(list_df):
        i = i+1
        print (f"*******   {i}/{delka}, runtime {time.time()-start} seconds ", end="\r")
        ans[_] = process_chunk(d)
    print(f"Finished in {round(time.time()-start)} seconds                           ")
    return ans

def main():
    mereni_df = df[["type", "tree", "day", "measurement"]].drop_duplicates().reset_index()
    process_df(mereni_df)

if __name__ == "__main__":
    main()






from dynatree.dynatree import DynatreeMeasurement
from dynatree.FFT import DynatreeSignal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from dynatree import find_measurements
from dynatree.dynatree import logger
import logging
import pandas as pd
import sys
from tqdm import tqdm

logger.setLevel(logging.INFO)

peak_min = .1 # do not look for the peak smaller than this value
peak_max = 0.7 # do not look for the peak larger than this value

def find_peak_width(m):
    s = DynatreeSignal(m, "Elasto(90)")

    frequencies = s.fft.index
    amplitudes = s.fft.values
    amplitudes[:8] = 0
    threshold = np.max(amplitudes)/np.sqrt(2)

    amax = np.max(amplitudes)
    treshold = amax/np.sqrt(2)
    _ = np.where(amplitudes - treshold > 0)[0]
    a = _[0]-1
    b = _[-1]+1


    def zero(a):
        return frequencies[a]- (amplitudes[a]-treshold)*(frequencies[a+1]-frequencies[a])/(amplitudes[a+1]-amplitudes[a])

    ans = (-zero(a) + zero(b-1))/amax

    fig, ax = plt.subplots()
    ax.plot(frequencies, amplitudes, "o-", label="FFT")
    plt.hlines(threshold, xmin=zero(a), xmax=zero(b-1), color='red', label="1/sqrt(2)*MAX")
    ax.set(yscale='log', ylim=(threshold/20, amax*2), xlim = (0.15,0.8),
           xlabel="freq/Hz", ylabel="Amplitude Elasto",
          title=f"{m}")
    ax.grid()
    ax.legend()
    return {'width':ans, 'figure':fig}

def process_row(row):
    date, tree, measurement, measurement_type, optics, day = row
    m = DynatreeMeasurement(day=day, tree=tree, measurement=measurement, measurement_type=measurement_type)
    return find_peak_width(m)


if __name__ == '__main__':

    import os
    import shutil

    source_dir = "figs_peak_width"
    # 1. Vymazání adresáře a jeho obsahu, pokud existuje
    if os.path.exists(source_dir):
        shutil.rmtree(source_dir)
        print(f"Adresář '{source_dir}' byl vymazán.")
    else:
        print(f"Adresář '{source_dir}' neexistoval, bude vytvořen nový.")

    # 2. Vytvoření adresáře
    os.makedirs(source_dir)
    print(f"Adresář '{source_dir}' byl vytvořen.")

    # 3. Naplnění daty
    try:
        matplotlib.use('TkAgg')
    except:
        matplotlib.use('Agg')
    out = {}

    df_failed = pd.read_csv("csv/FFT_failed.csv")
    df_failed = df_failed[df_failed["probe"]=="Elasto(90)"]
    df_failed = ["_".join(i) for i in df_failed.values[:,:4]]


    df = find_measurements.get_all_measurements(method='all', type='all')
    df = df[df["measurement"] != "M01"]
    df = df[df["type"] != "noc"]
    df = df[df["type"] != "den"]

    pbar = tqdm(total=len(df))
    for i,row in df.iterrows():
        pbar.update(1)
        data = [row.iloc[3],*row.iloc[0:3].values]
        data = "_".join(data)
        if data in df_failed:
            logger.info(f"Skipping {data} -- FFT marked as failed")
            continue
        # if [row['type'],row['day'],row['tree'],row[]]
        try:
            ans = process_row(row)
            out[tuple(row[:4])] = ans['width']
            plt.savefig(f"figs_peak_width/{row['date']}_{row['type']}_{row['tree']}_{row['measurement']}.png")
            plt.close()
        except:
            print(f"Row {row} failed.")
        break

    pbar.close()
    df = pd.DataFrame(out, index=["width"]).T
    df.index.names = ("day","tree","measurement","type")
    df.to_csv("../outputs/peak_width.csv")

    # 4. Zazipovat obrazky

    zip_command = f"zip -qr  peak_width.zip figs_peak_width"
    os.system(zip_command)
    shutil.move("peak_width.zip", "../outputs/peak_width.zip")

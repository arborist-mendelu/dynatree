from dynatree.dynatree import DynatreeMeasurement
from dynatree.FFT import DynatreeSignal

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from dynatree import find_measurements
from dynatree.dynatree import logger
import logging
import pandas as pd
from tqdm import tqdm
import sys
import config
from parallelbar import progress_map

import rich

logger.setLevel(logging.ERROR)

peak_min = .1 # do not look for the peak smaller than this value
peak_max = 0.7 # do not look for the peak larger than this value

def find_peak_width(m, save_fig=False, sensor="Elasto(90)"):
    """
    Return relative width of the peak (the width divided by the peak frequency)
    If save_fig is True, return a dictionary with a figure and peak width.
    If save_fig is False (default), return only a peak width.
    Otherwise the save_fig is assumed to be a name of the figure.

    :param m: Dynatree measurement object
    :param save_fig: True to return the figure, string to save the figure into a file
    :param sensor: sensor to use
    :return: the relative peak width and optionally a picture of the peak and the width.
    """
    if sensor in ["blueMaj", "yellowMaj"]:
        sensor = m.identify_major_minor[sensor]
    s = DynatreeSignal(m, signal_source=sensor)
    logger.info(f"Finding peak width. {m} {sensor}")
    try:
        fft = s.fft
    except:
        logger.error(f"Failed fft for {m} {sensor}")
        return None
    if fft is None:
        logger.error(f"Failed {m} {sensor}")
        return None

    frequencies = s.fft.index
    amplitudes = s.fft.values
    amplitudes[:10] = 0
    threshold = np.max(amplitudes)/np.sqrt(2)

    amax = np.max(amplitudes)
    fmax = frequencies[np.argmax(amplitudes)]
    treshold = amax/np.sqrt(2)
    _ = np.where(amplitudes - treshold > 0)[0]
    a = _[0]-1
    b = _[-1]+1

    def zero(a):
        return frequencies[a]- (amplitudes[a]-treshold)*(frequencies[a+1]-frequencies[a])/(amplitudes[a+1]-amplitudes[a])

    ans = (-zero(a) + zero(b-1))/fmax

    if save_fig == False:
        return ans

    fig, ax = plt.subplots()
    ax.plot(frequencies, amplitudes, "o-", label="FFT")
    plt.hlines(threshold, xmin=zero(a), xmax=zero(b-1), color='red', label="1/sqrt(2)*MAX")
    ax.set(yscale='log', ylim=(threshold/20, amax*2), xlim = (0.15,0.8),
           xlabel="freq/Hz", ylabel=f"Amplitude {sensor}",
          title=f"{m.day}, {m.measurement_type}, {m.tree}, {m.measurement}, {sensor}")
    ax.grid()
    ax.legend()

    if save_fig == True:
        return {'fig': fig, 'width': ans}
    else:
        plt.savefig(save_fig)
        plt.close()
    return ans

def process_row(row):
    day, measurement_type, tree, measurement, probes = row
    m = DynatreeMeasurement(day=day, tree=tree, measurement=measurement, measurement_type=measurement_type)
    if not m.is_optics_available:
        probes = [i for i in probes if "Pt" not in i]
    prefix = source_dir
    ans = [list(row[:-1])+ [
        sensor,
        find_peak_width(m, save_fig=f"{prefix}/peak_{m.day}_{m.measurement_type}_{m.tree}_{m.measurement}_{sensor}.png", sensor=sensor)
        ] for sensor in probes]
    return ans


if __name__ == '__main__':

    import os
    import shutil

    source_dir = "../temp/figs_peak_width"
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

    col_order = ["day","type", "tree", "measurement", "probe"]
    df_failed = pd.read_csv(config.file['FFT_failed'])
    df_failed = df_failed[col_order]

    df = find_measurements.get_all_measurements(method='all', type='all')
    df = df[df["measurement"] != "M01"]
    df = df.drop("optics", axis=1).drop_duplicates()[col_order[:-1]]

    probes = ["Elasto(90)", "blueMaj", "yellowMaj",
              "Pt3", "Pt4"] + [f"a0{i}_{j}" for i in [1, 2, 3, 4] for j in ["y","z"]]

    # Rozšíření tabulky df pro každou hodnotu 'probe'
    df_expanded = df.loc[df.index.repeat(len(probes))].copy()
    df_expanded['probe'] = probes * len(df)

    # Odstranění řádků, které se již nachází v df_failed
    df_result = df_expanded.merge(df_failed, how='left', on=['day', 'type', 'tree', 'measurement', 'probe'],
                                  indicator=True)
    df_result = (
        df_result[df_result['_merge'] == 'left_only'].
        drop(columns=['_merge']).groupby(['day', 'type', 'tree', 'measurement'], as_index=False)
        .agg({'probe': list})
    )
    df = df_result.copy()

    out = progress_map(process_row, [i for _,i in df.iterrows()], chunk_size=100)
    out = sum(out,[])
    out_df = pd.DataFrame(out, columns=["day", "type", "tree", "measurement", "probe", "width"])
    out_df.to_csv(config.file['outputs/peak_width'], index=False)

    # 4. Zazipovat obrazky

    zip_command = f"zip -qrj  peak_width.zip ../temp/figs_peak_width"
    os.system(zip_command)
    shutil.move("peak_width.zip", "../outputs/peak_width.zip")

"""
Ulozi fft obrazky a obrazky kmitu do souborove cache.
"""

import dynatree.signal_knock as sk
from dynatree.dynatree import DynatreeMeasurement
from dynatree.find_measurements import get_all_measurements_acc
from tqdm import tqdm
import pandas as pd
from config import file
import dynatree.dynatree as dynatree
import  logging
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import ChainMap
# import resource
# resource.setrlimit(resource.RLIMIT_AS, (10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024))

delta_time = 0.4


def process_row(row):
    ans = {}
    date, tree, measurement, type = row
    m = DynatreeMeasurement(date, tree, measurement, type)
    for knock_times, probes in zip(
            [sk.find_peak_times_channelA(m), sk.find_peak_times_channelB(m)],
            [sk.channelA, sk.channelB]
    ):
        for knock_index, knock_time in enumerate(knock_times):
            for probe in probes:
                coords = (type, date, tree, measurement, knock_index, int(knock_time * 100), probe)
                signal_knock = sk.SignalTuk(m, start=knock_time - delta_time, end=knock_time + delta_time, probe=probe)
                fft_peak = signal_knock.fft.iloc[5:].idxmax()
                figname = f"{type}_{date}_{tree}_{measurement}_{probe}_{int(knock_time * 100)}"
                ans[coords] = [fft_peak, figname]
                dynatree.logger.info(f"{type} {date} {tree} {measurement} {probe} {knock_time} {fft_peak}")
                del signal_knock
    del m
    plt.close('all')
    return ans

def process_row_safe(row):
    try:
        return process_row(row)
    except Exception as e:
        print(f"Chyba při zpracování {row}: {e}")
        return {}

def main():
    all_data = get_all_measurements_acc().drop_duplicates()
    pbar = tqdm(total=len(all_data))

    dynatree.logger.setLevel(logging.ERROR)

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(process_row_safe, row) for row in all_data.values]

        combined_dict = ChainMap()  # Inicializace ChainMap

        # Sledování průběhu
        for future in tqdm(as_completed(futures), total=len(futures), desc="Zpracovávám"):
            vysledek = future.result()  # Získání výsledku
            combined_dict = combined_dict.new_child(vysledek)  # Přidání slovníku do ChainMap

    # Kombinace ChainMap do jednoho slovníku
    ans = dict(combined_dict)

    df = pd.DataFrame(ans, index=["freq", "filename"]).T
    df = df.reset_index()
    df.columns = ["type", "day", "tree", "measurement", "knock_index", "knock_time", "probe", "freq", "filename"]
    df.to_csv("FFT_acc_knock_auto.csv", index=False)

if __name__ == "__main__":
    main()






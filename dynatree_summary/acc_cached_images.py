"""
Ulozi fft obrazky a obrazky kmitu do souborove cache.
"""
import sys

from tqdm import tqdm
import pandas as pd
from config import file
import matplotlib.pyplot as plt

delta_time = 0.4
df = pd.read_csv(file['outputs/FFT_acc_knock'])
df = df[df["type"]!="testovaci"]

from dynatree.dynatree import DynatreeMeasurement
from dynatree.signal_knock import SignalTuk

def save_images(signal_knock, fft_peak, figname):
    cachedir = file['cachedir']
    cachedir_large = file['cachedir_large']

    # small time domain
    fig, ax = plt.subplots(figsize=(3,1))
    ax.plot(signal_knock.signal)
    fig.savefig(f"{cachedir}/{figname}.png", transparent=True)
    # print(f"{cachedir}/{figname}.png saved")
    plt.close(fig)

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
    plt.close(fig)

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
    plt.close(fig)

    plt.close('all')


def zpracuj_mereni(row):
    columns_to_match = ["type", "tree", "day", "measurement"]
    subdf = df[(df[columns_to_match] == row[columns_to_match]).all(axis=1)]
    m = DynatreeMeasurement(day=row["day"], tree=row["tree"], measurement=row["measurement"],
                            measurement_type=row["type"])
    for subi,subrow in subdf.iterrows():
        s = SignalTuk(m, subrow["knock_time"] / 100.0 - delta_time,
                      subrow["knock_time"] / 100.0 + delta_time,
                      subrow["probe"])
        save_images(s, subrow['freq'], subrow['filename'])
        print (subrow['filename'])

def main():
    pbar = tqdm(total=len(df)
    for i, row in df.iterrows():
        zpracuj_mereni(row)
        pbar.update(1)
    pbar.close()

if __name__ == "__main__":
    main()






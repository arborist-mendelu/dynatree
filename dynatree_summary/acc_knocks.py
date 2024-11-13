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

# import resource
# resource.setrlimit(resource.RLIMIT_AS, (10 * 1024 * 1024 * 1024, 10 * 1024 * 1024 * 1024))

all_data = get_all_measurements_acc()
pbar = tqdm(total=len(all_data))
delta_time = 0.4
cachedir = file['cachedir']

dynatree.logger.setLevel(logging.ERROR)

def save_images(signal_knock, fft_peak, figname):

    fig, ax = plt.subplots(figsize=(3,0.8))
    ax.plot(signal_knock.signal)
    fig.savefig(f"{cachedir}/{figname}.png", transparent=True)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3,1))
    ax.plot(signal_knock.fft)
    fig.savefig(f"{cachedir}/FFT_{figname}.png", transparent=True)
    plt.close(fig)

    plt.close('all')


def main():
    ans = {}
    for row in all_data.values:
        date, tree, measurement, type = row
        m = DynatreeMeasurement(date, tree, measurement, type)
        for knock_times,probes in zip(
                [sk.find_peak_times_chanelA(m), sk.find_peak_times_chanelB(m)],
                [sk.chanelA, sk.chanelB]
        ):
            for knock_index, knock_time in enumerate(knock_times):
                for probe in probes:
                    coords = (type, date, tree, measurement, knock_index, int(knock_time*100), probe)
                    try:
                        signal_knock = sk.SignalTuk(m, start=knock_time - delta_time, end=knock_time + delta_time, probe=probe)
                        fft_peak = signal_knock.fft.iloc[5:].idxmax()
                        figname = f"{type}_{date}_{tree}_{measurement}_{probe}_{int(knock_time*100)}"
                        save_images(signal_knock, fft_peak, figname)
                        ans[coords] = [fft_peak, figname]
                        dynatree.logger.info(f"{type} {date} {tree} {measurement} {probe} {knock_time} {fft_peak}")
                    except:
                        ans[coords] = [None, None]
            #     break
            # break
        del m
        pbar.update(1)
        # break

    pbar.close()
    df = pd.DataFrame(ans, index=["freq", "filename"]).T
    df = df.reset_index()
    df.columns = ["type", "day", "tree", "measurement", "knock_index", "knock_time", "probe", "freq", "filename"]
    df.to_csv(file['outputs/FFT_acc_knock'], index=False)

main()






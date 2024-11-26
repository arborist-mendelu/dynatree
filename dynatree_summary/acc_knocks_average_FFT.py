from dynatree.dynatree import  DynatreeMeasurement, logger, logger_level
from dynatree.signal_knock import SignalTuk
from dynatree.find_measurements import get_all_measurements_acc
import config
import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib
from parallelbar import progress_map

from dynatree.signal_knock import SignalTuk
from dynatree.dynatree import DynatreeMeasurement, logger
from dynatree.find_measurements import get_all_measurements_acc
import logging

logger.setLevel(logging.WARNING)

CACHE = "../outputs/cache_FFTavg/"

def get_one_fft(m=None, **kwds):
    if m is None:
        logger.info(f"Loading {m}")
        m = DynatreeMeasurement(day=kwds['day'], tree=kwds['tree'], measurement=kwds['measurement'], measurement_type=kwds['measurement_type'])
    s = SignalTuk(m, start=round(kwds['knock_time']/100.0 - 0.04, 2), end=round(kwds['knock_time']/100.0 + 0.04, 2), probe=kwds['probe'])
    return s.fft

def get_one_tree(**kwds):
    measurement_type, day, tree = kwds['measurement_type'], kwds['day'], kwds['tree']
    ms = df[(df[["measurement_type","day","tree"]]==[measurement_type, day, tree]).all(axis=1)]['measurement'].drop_duplicates().to_list()
    m = {i: DynatreeMeasurement(day=day, tree=tree, measurement=i, measurement_type=measurement_type) for i in ms}
    fft_table = df[ (df[["measurement_type","day","tree"]] == [measurement_type, day, tree]).all(axis=1)].copy()
    fft_table = fft_table.set_index(["measurement", "knock_time", "probe"], drop=False)
    ans = fft_table.apply(lambda row:get_one_fft(m=m[row['measurement']],**row), axis=1)
    return ans.T


def process_one_tree_to_images(**kwds):
    logger.info(f"{kwds}")

    figsize = (10, 5)

    ans = get_one_tree(**kwds)
    probes = [f"a0{i}_{j}" for i in [1, 2, 3, 4] for j in ["x", "y", "z"]]

    for probe in probes:
        figname = f"{CACHE}FFTaverage_{kwds['measurement_type']}_{kwds['day']}_{kwds['tree']}_{probe}.png"
        csvname = f"{CACHE}FFTaverage_{kwds['measurement_type']}_{kwds['day']}_{kwds['tree']}_{probe}.csv"
        fig, ax = plt.subplots(figsize=figsize)
        try:
            subans = ans.xs(key=probe, level="probe", axis=1).sort_index(axis=1)
            subans.plot(linewidth=1, legend=None, alpha=0.3, ax=ax)

            data = subans.apply(lambda row: row.median(), axis=1)
            data.plot(color='red', ax=ax)

            ax.set(yscale='log')
            ax.set(
                title=f"{kwds['measurement_type']} {kwds['day']} {kwds['tree']} {probe}, data from {subans.shape[1]} curves")

            # Nastavení mřížky po 100
            ax.xaxis.set_minor_locator(plt.MultipleLocator(100))
            ax.grid(which='major', axis='x', color='gray', alpha=0.7, linewidth=2)
            ax.grid(which='major', axis='y', color='gray', alpha=0.7)
            ax.grid(which='both', axis='x', color='gray', alpha=0.7)
        except:
            subans = pd.DataFrame()
            ax.set(title=f"{kwds['measurement_type']} {kwds['day']} {kwds['tree']} {probe}, no data")
            logger.info(f"No curve for {probe}")

        ax.set(xlabel="Frequency / Hz", ylabel="FFT")
        # ax.get_legend().remove()
        plt.tight_layout()
        fig.savefig(figname, transparent=True)
        subans.to_csv(csvname)
        plt.close('all')

def process_wrapper(row):
    logger.info(f"Processing {row} by process_wrapper")
    process_one_tree_to_images(**row)

df = pd.read_csv(config.file['outputs/FFT_acc_knock'])
df = df[df["valid"] == True]
df = df.drop(["filename", "valid", "knock_index", "freq"], axis=1)
df = df.rename({"type": "measurement_type"}, axis=1)

list_of_measurements = df[["measurement_type", "day", "tree"]].drop_duplicates()

def main():
    try:
        matplotlib.use('TkAgg')  # https://stackoverflow.com/questions/39270988/ice-default-io-error-handler-doing-an-exit-pid-errno-32-when-running
    except:
        matplotlib.use('Agg')

    # print(list_of_measurements)
    progress_map(process_wrapper, [i for _, i in list_of_measurements.iterrows()], n_cpu=6)

if __name__ == "__main__":
    # process_one_tree_to_images(**{'day': '2024-09-02', 'tree': 'BK01', 'measurement_type': 'afterro2', 'probe': 'a04_y'})
    # process_one_tree_to_images(**{'day': '2022-08-16', 'tree': 'BK07', 'measurement_type': 'normal', 'probe': 'a04_y'})
    # get_one_tree(**{'day': '2022-08-16', 'tree': 'BK07', 'measurement_type': 'normal', 'probe': 'a04_y'})
    main()

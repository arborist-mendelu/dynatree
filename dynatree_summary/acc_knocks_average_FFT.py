from dynatree.dynatree import  DynatreeMeasurement, logger, logger_level
from dynatree.signal_knock import SignalTuk
from dynatree.find_measurements import get_all_measurements_acc
import config
import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib
from parallelbar import progress_map

logger.setLevel(logging.INFO)
CACHE = "../outputs/cache_FFTavg/"

def get_FFT_all_acc(**kwds):
    logger.info(f"get_FFT_all_acc entered, {kwds}")
    probes = [f"a0{i}_{j}" for i in [1,2,3,] for j in ["x","y","z"]]
    for probe in probes:
        try:
            get_FFT_one_probe(probe=probe, **kwds)
        except:
            logger.error(f"Failed get_FFT_one_probe for {kwds} and {probe}.")


def get_FFT_one_probe(**kwds):
    logger.info(f"get_FFT_one_probe entered, {kwds}")
    figname = f"{CACHE}FFTaverage_{kwds['type']}_{kwds['day']}_{kwds['tree']}_{kwds['probe']}.png"
    csvname = f"{CACHE}FFTaverage_{kwds['type']}_{kwds['day']}_{kwds['tree']}_{kwds['probe']}.csv"
    figsize = (10, 5)
    columns_for_select = ["type", "day", "tree", "probe"]
    subdf = df[(df[columns_for_select] ==
                [kwds[i] for i in columns_for_select]).all(axis=1)].copy()
    tuky = {}
    subdf_iter = pd.DataFrame(subdf.groupby('measurement')['knock_time'].agg(list))
    if len(subdf_iter)==0:
        logger.warning(f"No data for get_FFT_one_probe, {kwds}, return empty image and empty dataframe")
        fig, ax = plt.subplots(figsize=figsize)
        ax.set(title=f"{kwds['type']} {kwds['day']} {kwds['tree']} {kwds['probe']}, data missing")
        fig.savefig(figname, transparent=True)
        ans = pd.DataFrame()
        ans.to_csv(csvname)
        return None
    for i, l in subdf_iter.iterrows():
        m = DynatreeMeasurement(day=kwds['day'], tree=kwds['tree'], measurement_type=kwds['type'],
                            measurement=i)
        logger.info(m)
        for j in l.iloc[0]:
            s = SignalTuk(m, start=round(j / 100.0 - 0.04, 2), end=round(j / 100.0 + 0.04, 2), probe=kwds['probe'])
            tuky[(i, j)] = s.fft
            del s
        del m

    ans = pd.DataFrame(tuky)
    pocet = ans.shape[1]
    ans[['median', 'std']] = ans.apply(lambda row: [row.median(), row.std()], axis=1).to_list()
    fig, ax = plt.subplots(figsize=figsize)
    ans.plot(linewidth=1, legend=None, alpha=0.2, ax=ax)
    ans.loc[:, ["median"]].plot(ax=ax, color='red')
    ax.set(yscale='log', title=f"{kwds['type']} {kwds['day']} {kwds['tree']} {kwds['probe']}, data from {pocet} curves",
           xlabel="Frequency / Hz", ylabel="FFT")
    # Nastavení mřížky po 100
    ax.xaxis.set_minor_locator(plt.MultipleLocator(100))
    ax.grid(which='major', axis='x', color='gray', alpha=0.7, linewidth=2)
    ax.grid(which='major', axis='y', color='gray', alpha=0.7)
    ax.grid(which='both', axis='x', color='gray', alpha=0.7)

    ax.get_legend().remove()
    plt.tight_layout()
    fig.savefig(figname, transparent=True)
    ans.to_csv(csvname)
    plt.close('all')
    return ans

df = pd.read_csv(config.file['outputs/FFT_acc_knock'])
df = df[df["valid"] == True]
def main():
    try:
        matplotlib.use('TkAgg')  # https://stackoverflow.com/questions/39270988/ice-default-io-error-handler-doing-an-exit-pid-errno-32-when-running
    except:
        matplotlib.use('Agg')

    dfm = (get_all_measurements_acc()
        .drop(["measurement"], axis=1)
        .drop_duplicates()
        .reset_index(drop=True)
        .rename({'date': 'day'}, axis=1) )

    res = progress_map(get_FFT_all_acc_wrapper, [i for _, i in dfm.iterrows()],
                       n_cpu=5)

def get_FFT_all_acc_wrapper(i):
    get_FFT_all_acc(**i)

if __name__ == "__main__":
    # get_FFT_all_acc(**{'day': '2022-08-16', 'tree': 'BK13', 'type': 'normal'})
    # get_FFT_all_acc(**{'day': '2021-06-29', 'tree': 'BK08', 'type': 'normal'})
    get_FFT_one_probe(**{'day': '2021-06-29', 'tree': 'BK10', 'type': 'normal', 'probe': 'a02_y'})
    # get_FFT_one_probe(**{'day': '2024-09-02', 'tree': 'BK01', 'type': 'afterro2', 'probe': 'a04_y'})
    # main()

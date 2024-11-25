from dynatree.dynatree import  DynatreeMeasurement, logger, logger_level
from dynatree.signal_knock import SignalTuk
from dynatree.find_measurements import get_all_measurements_acc
import config
import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib
from parallelbar import progress_map

logger.setLevel(logging.ERROR)
CACHE = "../outputs/cache_FFTavg/"

def get_FFT_all_acc(**kwds):
    logger.info(f"get_FFT_all_acc entered, {kwds}")
    probes = [f"a0{i}_{j}" for i in [1,2,3,] for j in ["x","y","z"]]
    for probe in probes:
        try:
            get_FFT_one_probe(probe=probe, **kwds)
        except:
            logger.error(f"Failed get_FFT_all_acc for {kwds}")


def get_FFT_one_probe(**kwds):
    logger.info(f"get_FFT_one_probe entered, {kwds}")
    subdf = df[(df[["type", "day", "tree", "probe"]] ==
                [kwds['type'], kwds['day'], kwds['tree'], kwds['probe']]).all(axis=1)].copy()
    tuky = {}
    for i, l in pd.DataFrame(subdf.groupby('measurement')['knock_time'].agg(list)).iterrows():
        m = DynatreeMeasurement(day=kwds['day'], tree=kwds['tree'], measurement_type=kwds['type'],
                            measurement=i)
        for j in l.iloc[0]:
            s = SignalTuk(m, start=round(j / 100.0 - 0.04, 2), end=round(j / 100.0 + 0.04, 2), probe=kwds['probe'])
            tuky[(i, j)] = s.fft
            del s
        del m

    ans = pd.DataFrame(tuky)
    ans[['median', 'std']] = ans.apply(lambda row: [row.median(), row.std()], axis=1).to_list()

    fig, ax = plt.subplots(figsize=(10, 5))
    ans.plot(linewidth=1, legend=None, alpha=0.2, ax=ax)
    ans.loc[:, ["median"]].plot(ax=ax, color='red')
    ax.set(yscale='log', title=f"{kwds['type']} {kwds['day']} {kwds['tree']} {kwds['probe']}")
    ax.grid()
    plt.tight_layout()
    fig.savefig(f"{CACHE}FFTaverage_{kwds['type']}_{kwds['day']}_{kwds['tree']}_{kwds['probe']}.png")
    ans.to_csv(f"{CACHE}FFTaverage_{kwds['type']}_{kwds['day']}_{kwds['tree']}_{kwds['probe']}.csv")
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
                       #executor='threads', 
                       n_cpu=5)
    # for _, i in dfm.iterrows():
    #     get_FFT_all_acc_wrapper(i)
    #     print(f"{_} finished")

def get_FFT_all_acc_wrapper(i):
    get_FFT_all_acc(**i)

if __name__ == "__main__":
    main()

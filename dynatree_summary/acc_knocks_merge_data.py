import pandas as pd

from dynatree.dynatree import DynatreeMeasurement, logger
from dynatree.signal_knock import SignalTuk

df_manual_failed = pd.read_csv("./dynatree_summary/FFT_acc_knock_fail_manual.csv",
                               index_col=None, header=None,
                               names=["filename"])
df_manual_failed["valid"] = False
df_manual_failed=df_manual_failed.set_index("filename")
df_manual_failed

#%%
df = pd.read_csv("./dynatree_summary/FFT_acc_knock_auto.csv", index_col=None)
df["valid"] = True
df = df.set_index("filename")

#%%
df.update(df_manual_failed)
df.to_csv("dynatree_summary/FFT_acc_knock.csv")

#%%
df_new = pd.read_csv("./dynatree_summary/FFT_acc_knock_peak_manual.csv", index_col=None, sep=',', skipinitialspace=True)
df_new['values'] = df_new['values'].str.strip("[]").str.split(", ").apply(lambda x: list(map(float, x)))
row = df_new.iloc[0,:]
row
#%%


def process_one_row(row):
    logger.debug(f"process one row entered with argument {row}")
    out = {}
    probe = row['probe']
    if len(probe)>3:
        probes = [probe]
    else:
        probes = [f"{probe}_{i}" for i in ["x", "y", "z"] ]
    for knock_time in row['values']:
        m = DynatreeMeasurement(day=row['day'], tree=row['tree'], measurement=row['measurement'], measurement_type=row['measurement_type'])
        for probe in probes:
            logger.debug(f"measurement {m}, probe {probe}")
            s = SignalTuk(m, start=round(knock_time-0.04,2), end=round(knock_time+0.04,2), probe=probe)
            maxfft = s.fft.idxmax()
            knock_time_int = round(knock_time*100)
            filename = f"{m.measurement_type}_{m.day}_{m.tree}_{m.measurement}_{probe}_{knock_time_int}"
            out[filename] = [m.measurement_type, m.day, m.tree, m.measurement, None, knock_time_int, probe, maxfft, True]

    out = pd.DataFrame(out).T
    out.columns = ["type","day","tree","measurement","knock_index","knock_time","probe","freq","valid"]
    return out

d = df_new.apply(process_one_row, axis=1).to_list()

df = pd.concat([df,*d],axis=0)
df.index.name = 'filename'
df.to_csv("dynatree_summary/FFT_acc_knock.csv")






import dynatree.signal_knock as sk
from dynatree.dynatree import DynatreeMeasurement
from dynatree.find_measurements import get_all_measurements_acc
from tqdm import tqdm
import pandas as pd
from config import file
import dynatree.dynatree as dynatree
import  logging

all_data = get_all_measurements_acc()
pbar = tqdm(total=len(all_data))

dynatree.logger.setLevel(logging.ERROR)

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
                try:
                    signal_knock = sk.SignalTuk(m, start=knock_time - 0.4, end=knock_time + 0.4, probe=probe)
                    fft_peak = signal_knock.fft.iloc[5:,:].idxmax()
                    ans[(date, tree, measurement, type, knock_index, probe)] = fft_peak
                    dynatree.logger.info(f"{type} {date} {tree} {measurement} {probe} {knock_time} {fft_peak}")
                except:
                    ans[(date, tree, measurement, type, knock_index, probe)] = None
    pbar.update(1)

pbar.close()

df = pd.DataFrame(ans, index=["Freq"]).T
df = df.reset_index()
df.columns = ["day", "tree", "measurement", "type", "knock_index", "probe", "freq"]
df.to_csv(file['outputs/FFT_acc_knock'], index=False)

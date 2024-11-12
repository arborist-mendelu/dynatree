from dynatree.signal_knock import find_peak_times, SignalTuk
from dynatree.dynatree import DynatreeMeasurement
from dynatree.find_measurements import get_all_measurements_acc
from tqdm import tqdm
import pandas as pd
from config import file

all_data = get_all_measurements_acc()
pbar = tqdm(total=len(all_data))

ans = {}
for row in all_data.values:
    date, tree, measurement, type = row
    m = DynatreeMeasurement(date, tree, measurement, type)
    knock_times = find_peak_times(m)
    probes = [f"a0{i}_{j}" for i in [1, 2, 3, 4] for j in ['x', 'y', 'z']]
    for knock_index, knock_time in enumerate(knock_times):
        for probe in probes:
            try:
                signal_knock = SignalTuk(m, start=knock_time - 0.4, end=knock_time + 0.4, probe=probe)
                fft_peak = signal_knock.fft.idxmax()
                ans[(date, tree, measurement, type, knock_index, probe)] = fft_peak
            except:
                ans[(date, tree, measurement, type, knock_index, probe)] = None
    pbar.update(1)

pbar.close()

df = pd.DataFrame(ans, index=["Freq"]).T
df = df.reset_index()
df.columns = ["day", "tree", "measurement", "type", "knock_index", "probe", "freq"]
df.to_csv(file['outputs/FFT_acc_knock'], index=False)

import pandas as pd
import numpy as np
from dynatree.signal_knock import SignalTuk
from dynatree.dynatree import DynatreeMeasurement

df = pd.read_csv("https://um.mendelu.cz/dynatree/static/public/FFT_acc_knock.csv")


def get_ratio(row):
    m = DynatreeMeasurement(day=row['day'], tree=row['tree'], measurement=row['measurement'], measurement_type=row['type'])
    s = SignalTuk(m, start=row['knock_time']/100.0-0.4, end=row['knock_time']/100.0+0.4, probe=row['probe'])
    initial = s.signal.iloc[:50]
    position = np.argmax(s.signal.index > row['knock_time']/100)
    bump = s.signal.iloc[position-5:position+45]
    return bump.abs().mean()/initial.abs().mean()

for i in range(0,(len(df)//100)+1):
    tempdf = df.loc[i*100-1:(i+1)*100,:].copy()
    tempdf["podil"] = tempdf.apply(get_ratio, axis=1)
    tempdf.to_csv(f"sila_uderu_{i:05}.csv")


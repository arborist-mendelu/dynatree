# %%

from dynatree.find_measurements import get_all_measurements
import pandas as pd
from dynatree.static_pull import DynatreeStaticMeasurement

df = get_all_measurements(method='all', type='all')
df = df[
    df["day"].isin(["2021-03-22", "2021-06-29"])
    & df['tree'].str.contains("BK")
    & df['measurement'].str.contains("M01")
    ]

df

# %%

data = {}
for _, row in df.iloc[:, :].iterrows():
    m = DynatreeStaticMeasurement(day=row['date'], tree=row['tree'], measurement=row['measurement'],
                                  measurement_type=row['type'], restricted=None)
    for i, s in enumerate(m.pullings):
        data[m.measurement_type, m.day, m.tree, m.measurement, i] = s.data['Force(100)'].max()

# %%

ans = pd.DataFrame(data, index=["Fmax"]).T
ans
# %%

grouped = ans.groupby(level=[0, 1, 2, 3]).max()
grouped.to_csv("F_max.csv")
grouped

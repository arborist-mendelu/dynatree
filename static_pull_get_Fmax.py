#%%

from lib_find_measurements import get_all_measurements
import static_pull
import rich
import matplotlib.pyplot as plt
import pandas as pd
from static_pull import DynatreeStaticMeasurement

df = get_all_measurements(method='all', type='all')


#%%

data = {}
for _,row in df.iloc[:,:].iterrows():
    m = DynatreeStaticMeasurement(day=row['date'], tree=row['tree'], measurement=row['measurement'], measurement_type=row['type'], restricted=None)
    for i,s in enumerate(m.pullings):
        data[m.measurement_type,m.day,m.tree,m.measurement,i] = s.data['Force(100)'].max()

#%%

ans = pd.DataFrame(data, index=["Fmax"]).T







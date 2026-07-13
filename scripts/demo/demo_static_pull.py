# %%
import os
os.environ["PREFIX_DYNATREE"] = "/home/marik/dynatree/scripts/"
os.environ["DYNATREE_DATAPATH"] = "/home/marik/dynatree/data/"
import sys
import pandas as pd
sys.path.append("..")
from dynatree.static_pull import DynatreeStaticMeasurement, DynatreeStaticPulling
from dynatree.static_pull import proces_one_row

# %%
day = "2022-08-16"
tree = "BK04"
measurement = "M01"
measurement_type = "normal"
data_obj = DynatreeStaticMeasurement(day=day, tree=tree, measurement=measurement, measurement_type=measurement_type)

proces_one_row({'day': day, 'tree': tree, 'measurement': measurement, 'measurement_type': measurement_type, 'optics': False, 'type': 'normal'})


# %%

data_obj.pullings[0].regressions
# %%

data_obj.pullings[0].data.blueMaj.plot()

# %%

def full_pullings(self):
    ans = [
        DynatreeStaticPulling(i, tree=self.tree, 
                measurement_type=self.measurement_type, day=self.day,
                parent_experiment=self,
                extra_columns={"blue":"Inclino(80)", "yellow":"Inclino(81)",
                **self.identify_major_minor}) 
        for i in self._get_static_pulling_data(
            optics=self.optics, restricted=self.restricted)
        ]
    return ans

times = data_obj._split_df_static_pulling()
times 

# %%
import matplotlib.pyplot as plt
df = data_obj.data_pulling[["Force(100)"]]

for i in times:
    df = data_obj.data_pulling[["Force(100)"]]
    # plot fot time range between i['minimum'] and i['maximum']
    df = df[(df.index >= i['minimum']) & (df.index <= i['maximum'])]
    plt.plot(df.index, df["Force(100)"])
# %%

proces_one_row({'day': day, 'tree': tree, 'measurement': measurement, 'measurement_type': measurement_type, 'optics': False, 'type': 'normal'})

# %%

import pandas as pd

df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'B': [5, 6], 'C': [7, 8]})

vysledek = pd.concat([df1, df2])
print(vysledek)
# %%


import pandas as pd

df = pd.DataFrame({
    'Produkt': ['A', 'B', 'A', 'C'],
    'Množství': [10, 20, 30, 40]
})

# 1. Založíme sloupec a naplníme ho prázdnou hodnotou (None)
# 2. Tam, kde je 'Produkt' roven 'A', zapíšeme do sloupce 'cena' hodnotu 10
df.loc[df["Produkt"] == "A", "cena"] = 10
df
# %%

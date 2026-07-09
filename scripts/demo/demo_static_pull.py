# %%
import os
os.environ["PREFIX_DYNATREE"] = "/home/marik/dynatree/scripts/"
os.environ["DYNATREE_DATAPATH"] = "/home/marik/dynatree/data/"
import sys
import pandas as pd
sys.path.append("..")
from dynatree.static_pull import DynatreeStaticMeasurement


# %%
day = "2022-08-16"
tree = "BK04"
measurement = "M01"
measurement_type = "normal"
data_obj = DynatreeStaticMeasurement(day=day, tree=tree, measurement=measurement, measurement_type=measurement_type)

# %%

data_obj.pullings[0].regressions
# %%

data_obj.pullings[0].data

# %%

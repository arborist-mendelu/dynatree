# %%
import os
os.environ["PREFIX_DYNATREE"] = "/home/marik/dynatree/scripts/"
os.environ["DYNATREE_DATAPATH"] = "/home/marik/dynatree/data/"
import sys
sys.path.append("..")

import dynatree.dynatree as dt
import dynatree.damping as dd
m = dt.DynatreeMeasurement(
    day="2021-03-22", 
    tree="BK12", 
    measurement_type="normal", 
    measurement="M02")

# %%

# %%
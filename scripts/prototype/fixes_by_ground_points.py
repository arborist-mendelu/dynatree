# %%
import sys
import os
os.environ["PREFIX_DYNATREE"] = "/home/marik/dynatree/scripts/"
os.environ["DYNATREE_DATAPATH"] = "/home/marik/dynatree/data/"
import numpy as np
import pandas as pd
sys.path.append("..")

# %%
import dynatree.dynatree as dtree
import dynatree.FFT as dfft

m = dtree.DynatreeMeasurement(day="2021-06-29", tree="BK04", measurement="M01")

# %%

df = m.data_optics[[(f"Pt{i}","Y0") for i in [3,4,11,12,13]]].copy()
df.columns = [i[0] for i in df.columns]
df = df - df.iloc[0,:]
df["Pt3_fixed"] = df["Pt3"] - df["Pt11"] 

df[["Pt3_fixed","Pt3"]].plot(subplots=True)


# %%


m1 = dtree.DynatreeMeasurement(day="2021-03-22", tree="BK21", measurement="M02")
m2 = dtree.DynatreeMeasurement(day="2021-03-22", tree="BK21", measurement="M04")

import matplotlib.pyplot as plt
df1 = m1.data_optics[[(f"Pt{i}","Y0") for i in [3]]].copy()
df2 = m2.data_optics[[(f"Pt{i}","Y0") for i in [3]]].copy()


# %%
fig, ax = plt.subplots(2,1)
df1.plot(ax=ax[0])
df2.plot(ax=ax[1])

# %%

d = df1 - df2
d.plot()

# %%

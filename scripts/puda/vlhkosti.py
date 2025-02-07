import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# načtení dat
df = pd.read_csv("../../outputs/vlhkosti_babice.csv")
df = df [["vzorek","hmotnostní vlhkost  w", "den"]]
# ve soupci vzorek rozdelit zapisy typu 1A na dva sloupce, sonda a horizont.
df["horizont"] = df["vzorek"].apply(lambda x: x[-1])
df["sonda"] = df["vzorek"].apply(lambda x: x[:-1])

# načtení tabulky pro převod mezi stromy a sondami
sondy = pd.read_csv("../../outputs/sondy_a_stromy.csv")


print(sondy)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# načtení dat
df = pd.read_csv("../../outputs/penetrologger.csv")

# Vyber jmena sloupcu s ciselnymi hodnotami
cisla = list(df.select_dtypes(include=[np.number]).columns)
# prumer za vsechny stromy v jednotlivych hloubkach
group_df = df[["day","tree"] + cisla].groupby(["tree","day"]).mean()
# prumer za vsechny hloubky
group_df["mean"] = group_df[cisla].mean(axis=1)

prehled = group_df.pivot_table(index="day", columns="tree", values="mean")
print(prehled)
# vykresleni tabulky pomoci heatmap
sns.heatmap(prehled, cmap="viridis")
plt.show()
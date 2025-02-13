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
# vynechat radky obsahujici ve sloipci tree hodnotu BK25 a JD18
group_df = group_df[~group_df.index.get_level_values("tree").isin(["BK25", "JD18"])]

prehled = group_df.pivot_table(index="day", columns="tree", values="mean")
print(prehled)
# vykresleni tabulky pomoci heatmap
sns.heatmap(prehled, cmap="viridis")
plt.show()
# barplot pro jednotlive stromy, siroky obrazek
fig, ax = plt.subplots(figsize=(15,5))
sns.barplot(data=group_df.reset_index(), x="day", y="mean", hue="tree", ax=ax) 

# barplot pro jednotlive dny
sns.barplot(data=group_df.reset_index(), x="tree", y="mean", hue="day")

# prubeh hodnot jako funkce datumu, pro kazdy strom v samostatnem grafu
for tree in group_df.index.get_level_values("tree").unique():
    plt.figure()
    sns.lineplot(data=group_df.loc[tree].reset_index(), x="day", y="mean")
    plt.title(tree)
    plt.show()

# boxplot pro jednotlive dny, vsechny stromy dohromady
sns.boxplot(data=group_df.reset_index(), x="day", y="mean")

# načtení dat
df_v = pd.read_csv("vlhkosti_babice.csv")
df_v = df_v [["vzorek","hmotnostní vlhkost  w", "den"]]
df_v = df_v[df_v["den"] != "2022-08-16"]
# ve soupci vzorek rozdelit zapisy typu 1A na dva sloupce, sonda a horizont.
df_v["horizont"] = df_v["vzorek"].apply(lambda x: x[-1])
df_v["sonda"] = df_v["vzorek"].apply(lambda x: x[:-1])
fig, ax = plt.subplots(figsize=(15,5))
sns.boxplot(data=df_v, x="den", y="hmotnostní vlhkost  w", ax=ax)
ax.grid()

print(df_v.head())
print(group_df.head())



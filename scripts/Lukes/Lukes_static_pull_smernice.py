import pandas as pd
import numpy as np

df = pd.read_csv("../outputs/anotated_regressions_static.csv", index_col=0)
df = df[
    df['measurement'].str.contains("M01")
    & df["day"].isin(["2021-03-22", "2021-06-29"])
    & df['tree'].str.contains("BK")
    & ~df['failed']
    ]

df_pull = df.copy()
df_pull = df_pull[
    df_pull['measurement'].str.contains("M01")
    & ~df_pull['optics']
    & df_pull['Dependent'].isin(["blue", "yellow", "Elasto-strain"])
    ]
df_pull

# %%
df_optics = df.copy()
df_optics = df_optics[
    df_optics["Dependent"].isin(["Pt3", "Pt4"])
]
df_optics["Slope"] = np.abs(df_optics["Slope"])
df_optics

# %%
df_all = pd.concat([df_pull, df_optics])
df_all = df_all.pivot(index=["type", "day", "tree", "measurement", "pullNo"], columns=["Dependent"], values=["Slope"])

df_all.columns = [i[1] for i in df_all.columns]
df_all["inclinometers"] = df_all[["blue", "yellow"]].mean(axis=1, skipna=True)
df_all = df_all.drop(["blue", "yellow"], axis=1)
grouped = df_all.groupby(level=[0, 1, 2, 3]).median()
grouped.to_csv("Inclino_Extenso_Optics.csv")

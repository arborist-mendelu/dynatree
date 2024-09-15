#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 10:20:56 2024

@author: marik
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lib_dynatree
from scipy.stats import ttest_1samp
from scipy import stats


def add_leaves_info(df_):
    df = df_.copy()
    days_with_leaves_true = ["2021-06-29", "2021-08-03", "2022-08-16", "2023-07-17", "2024-09-02"]
    days_after_first_reduction = ['2024-01-16', '2024-04-10', '2024-09-02']
    
    # Set information about leaves.
    df.loc[:,"leaves"] = False
    idx = (df["day"].isin(days_with_leaves_true))
    df.loc[idx,"leaves"] = True
    # no reduction is default
    df.loc[:,"reductionNo"] = 0
    # one reduction: afterro or any day later
    idx = (df["day"].isin(days_after_first_reduction))
    df.loc[idx,"reductionNo"] = 1
    idx = (df["type"]=="afterro")
    df.loc[idx,"reductionNo"]=1
    # two reductions
    idx = (df["type"]=="afterro2")
    df.loc[idx,"reductionNo"]=2
    return df


df = pd.read_csv("../outputs/anotated_regressions_static.csv", index_col=0)
df = df.dropna(subset=["Independent","Dependent"],how='all')
df = df[df["lower_cut"]==0.3]
df = df.dropna(how='all', axis=0)
df = df[~df['Dependent'].str.contains('Min')]
df = df[~df['tree'].str.contains('JD')]
df = df[~ df["failed"]]
df = df.drop(["Intercept","p-value","stderr","intercept_stderr","lower_cut", "upper_cut"], axis=1)



a = df[(df["optics"]) & (df['Dependent'].str.contains('Pt'))]
b = df[~df["optics"]]
oridf = pd.concat([a,b]).reset_index(drop=True)
oridf

# # Je u prvního měření větší tuhost?

df = oridf.copy()
df = df[
    (df["measurement"]=="M01") & 
     (df["Independent"]=="M")
    ].drop(["measurement","optics"], axis=1)
df

# +
# Nejprve vytáhneme hodnoty Slope pro pull=0
df_zero_pull = df[df['pullNo'] == 0].copy()

# Přejmenujeme sloupec Slope, aby bylo jasné, že jde o referenční hodnoty
df_zero_pull = df_zero_pull.rename(columns={'Slope': 'Slope_zero_pull'})

# Merge původního DataFrame s tím, kde je pull=0, na základě společných sloupců
df_merged = pd.merge(df, df_zero_pull[['day', 'tree', 'Independent', 'Dependent', 'Slope_zero_pull']],
                     on=['day', 'tree', 'Independent', 'Dependent'], how='left')

# Vydělení hodnoty Slope referenční hodnotou Slope_zero_pull
for i in ['Slope']:
    df_merged[f'{i}_normalized'] = df_merged[i] / df_merged[f'{i}_zero_pull']
# -

df_merged

target = "Slope"
subdf = df_merged[df_merged["pullNo"]!=0].loc[:,["pullNo","Slope_normalized","tree"]]
fig, ax = plt.subplots(1,1,figsize=(20,5))
subdf['pullNo'] = subdf['pullNo'].astype('category')
sns.stripplot(data=subdf, x='tree', y=f'{target}_normalized', hue='pullNo', ax=ax, jitter=0.4)
ax.grid()
ax.set(ylim=(None,1.5))

# +
result1 = subdf.groupby('tree')['Slope_normalized'].agg(['mean', 'std']).reset_index()

# Funkce pro testování hypotézy
def test_hypothesis(group):
    t_stat, p_value = ttest_1samp(group['Slope_normalized'], 1)
    return pd.Series({'t_stat': t_stat, 'p_value': p_value})

# Aplikace funkce na skupiny dle kategorie 'tree'
result2 = subdf.groupby('tree').apply(test_hypothesis, include_groups=False).reset_index()

result = pd.merge(result1,result2, on=["tree"])
result["H0_rejected"] = result["p_value"]<0.05
result.sort_values(by="p_value").reset_index(drop=True)
# -

# # Je blueMaj stejné jako blue? Totéž pro yellow.

# +
color = "blue"
# color = "yellow"
df = oridf.copy()
df = (df[df["Dependent"].str.contains(color)]
        .drop(["R^2","optics"], axis=1)
        .pipe(add_leaves_info)
     )

def nakresli_inklino(row, i=""):
    m = lib_dynatree.DynatreeMeasurement(
        day=row['day'], tree=row['tree'], measurement=row['measurement'], 
        measurement_type=row['type']
    )
    ax = m.data_pulling.loc[:,["Inclino(80)X","Inclino(80)Y"]].plot()
    # ax = m.data_pulling.plot(
    #         y=m.identify_major_minor.values(),
    #         style='.', ms=2,
    #         )
    ax.grid()
    ax.set(title=f"{i} {m} {row['pullNo']}")

df_wide = df.pivot_table(
    index=['Independent', 'day', 'tree', 'measurement', 'type', 'pullNo', 'leaves', 'reductionNo'], 
    columns='Dependent', 
    values='Slope'
).reset_index()

df_wide["difference"] = df_wide[color] - df_wide[f"{color}Maj"]
df_wide["relative_difference"] = df_wide['difference'] / df_wide[f"{color}Maj"]
df_wide = df_wide.sort_values(by="relative_difference").reset_index(drop=True)
df_wide.to_csv("rozdily_blue.csv")
df_wide
# -

# Testujeme, jestli ve sloupci "blue" a "blueMaj" jsou stejne hodnoty. Nulova hypoteza je, ze rozdil mezi sloupci blue a blueMaj je nula. Pokud je p_value mensi nez 0.05, potom pravdepodobnost, ze by se jednalo o vybery ze souboru se stejnou stredni hodnotou je mensi nez pet procent a nulova hypoteza se zamita. To znamena, ze mezi blue a blueMaj je rozdil. 
#
# Pokud je vetsi, nemuzeme zamitnout nulovou hypotezu a predpokladame, ze mezi blue a blueMaj neni rozdil.
# Ale jeste je potreba zkontrolovat normalitu!!!! Ta bohuzel neni!!!

df_wide = df_wide.dropna()
t_stat, p_value = stats.ttest_rel(df_wide[color], df_wide[f"{color}Maj"])
print (p_value)

# ### Nejvíc odlišná měření
#
# Deset měření, kde je blue a blueMajor nejvíc liší na jednu a na druhou stranu. Číslo na konci nadpisu je pořadí zatáhnutí (má smysl jenom u M01).

for i,row in pd.concat([df_wide.tail(10),df_wide.head(10)]).iterrows():
    nakresli_inklino(row,i)

ax = df_wide.relative_difference.plot(style='.')
ax.set(title=f"relativni rozdíl mezi {color} a {color}Maj")
ax.grid()

stat, p_value = stats.shapiro(df_wide['difference'])
p_value  # Musi byt vetsi nez 0.05, aby se dalo tvrdit, ze data jsou normalne rozlozena

out = stats.probplot(df_wide['difference'], dist="norm", plot=plt)

plt.hist(df_wide['difference'], bins=20, edgecolor='black')

# +
# Wilcoxonův párový test
stat, p_value = stats.wilcoxon(df_wide['blue'], df_wide['blueMaj'])
print(f"Wilcoxon signed-rank test: p-value = {p_value}")
# Mann-Whitney U test
stat, p_value = stats.mannwhitneyu(df_wide['blue'], df_wide['blueMaj'])
print(f"Mann-Whitney U test: p-value = {p_value}")
# Kruskal-Wallisův test
stat, p_value = stats.kruskal(df_wide['blue'], df_wide['blueMaj'])
print(f"Kruskal-Wallis test: p-value = {p_value}")
# Sign test (pomocí scipy není implementováno přímo, ale dá se vytvořit pomocí počítání kladných a záporných rozdílů)
# Sign test pomocí binomtest
n_successes = (df_wide['blue'] > df_wide['blueMaj']).sum()
n_trials = len(df_wide)

# Provedení binomtestu
result = stats.binomtest(n_successes, n=n_trials, p=0.5)
print(f"Sign test: p-value = {result.pvalue}")
# -

df_wide.loc[:,'relative_difference'].median(), df_wide.loc[:,'relative_difference'].std()



# # Je yellowMaj stejné jako yellow?

# +
color = "yellow"
df = oridf.copy()
df = (df[df["Dependent"].str.contains(color)]
        .drop(["R^2","optics"], axis=1)
        .pipe(add_leaves_info)
     )

def nakresli_inklino(row, i=""):
    m = lib_dynatree.DynatreeMeasurement(
        day=row['day'], tree=row['tree'], measurement=row['measurement'], 
        measurement_type=row['type']
    )
    ax = m.data_pulling.loc[:,["Inclino(81)X","Inclino(81)Y"]].plot()
    # ax = m.data_pulling.plot(
    #         y=m.identify_major_minor.values(),
    #         style='.', ms=2,
    #         )
    ax.grid()
    ax.set(title=f"{i} {m} {row['pullNo']}")

df_wide = df.pivot_table(
    index=['Independent', 'day', 'tree', 'measurement', 'type', 'pullNo', 'leaves', 'reductionNo'], 
    columns='Dependent', 
    values='Slope'
).reset_index()

df_wide["difference"] = df_wide[color] - df_wide[f"{color}Maj"]
df_wide["relative_difference"] = df_wide['difference'] / df_wide[f"{color}Maj"]
df_wide = df_wide.sort_values(by="relative_difference").reset_index(drop=True)
df_wide.to_csv("rozdily_yellow.csv")
df_wide
# -

# ### Nejvíc odlišná měření
#
# Deset měření, kde je blue a blueMajor nejvíc liší na jednu a na druhou stranu. Číslo na konci nadpisu je pořadí zatáhnutí (má smysl jenom u M01).

for i,row in pd.concat([df_wide.tail(10),df_wide.head(10)]).iterrows():
    nakresli_inklino(row,i)

ax = df_wide.relative_difference.plot(style='.')
ax.set(title=f"relativni rozdíl mezi {color} a {color}Maj")
ax.grid()

plt.hist(df_wide['difference'], bins=20, edgecolor='black')

df_wide.loc[:,'relative_difference'].median(), df_wide.loc[:,'relative_difference'].std()



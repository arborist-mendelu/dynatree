#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# +

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lib_dynatree
from scipy.stats import ttest_1samp
from scipy import stats

import static_lib_pull_comparison
# -

# # Je u prvního měření větší tuhost?
#
# Sledujeme statické měření. Určíme směrnice mezi veličinami, co nás zajímají. Je směrice v prvním tahu odlišná od směrnice ve druhém a dalším zatažení? Vydělíme
# druhou a třetí směrnici první směrnicí porovnáváme podíly s jedničkou. 
#
# Použitá data jsou regresní koeficienty ze závislostí M versus blue, M verus blueMajor, M verus yellow, M versus yellowMajor a M_Elasto verus Elasto-strain.
# Poslední závislost je klesající (stlačování extenzometru), proto je směrnice záporná. Ale po vydělení zápornou směrnicí z prvního zatáhnutí se to srovná.

df_merged = static_lib_pull_comparison.df_merged


# +

target = "Slope"
subdf = df_merged[df_merged["pullNo"]!=0].loc[:,["pullNo","Slope_normalized","tree"]]
fig, ax = plt.subplots(1,1,figsize=(20,5))
subdf['pullNo'] = subdf['pullNo'].astype('category')
sns.stripplot(data=subdf, x='tree', y=f'{target}_normalized', hue='pullNo', ax=ax, jitter=0.4)
ax.set(title="Podíl směrnic následujícího a prvního zatáhnutí")
ax.grid(axis='y')
ax.legend(bbox_to_anchor=(1.1, 1))
ax.axhline(1,color='red')
[ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]

# ax.legend(["druhé zatáhnutí", "třetí zatáhnutí", "čtvrté zatáhnutí"])
ax.set(ylim=(None,None));
# -

# Jestli jsou podíly rovny jedné nebo ne je možné otestovat i statistickým testem.

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
# # Srování uvnitř jednotlivých stromů
#
# Pro každý strom rozseparujeme data podle dnů.

# +
target == 'Slope'
trees = subdf["tree"].sort_values().drop_duplicates().tolist()

for tree in trees:
    fig, ax = plt.subplots(1,1,figsize=(15,5))
    subdf = df_merged[df_merged["tree"]==tree]
    subdf = subdf[subdf["pullNo"]!=0]
    sns.boxplot(data=subdf, x='day', y=f'{target}_normalized', hue='type', ax=ax, boxprops={'alpha': 0.4})
    sns.stripplot(data=subdf, x='day', y=f'{target}_normalized', hue='type', ax=ax, dodge=True)
    ax.grid(axis='y')
    ax.set(title=tree)
    ax.legend(bbox_to_anchor=(1.12, 1))
    ax.axhline(1,color='red')
    [ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]


# -


# Deset měření, kde je poměr mezi následným a prvním zatáhnutím nejdále od jedničky. Deset extrémů z každého konce. Zajímá nás sloupec "Slope normalized", který je 
# pro první zatáhnutí jedna a pro další zatáhnutí jsme věřili, že bude menší než jedna.

df_merged.sort_values(by="Slope_normalized").tail(10)


df_merged.sort_values(by="Slope_normalized").head(10)


# Průměrná hodnota, jaké procento tuhosti je při druhém a dalších zatáhnutích (100% je první zatáhnutí).

df_merged["Slope_normalized"].mean()

# +

import plotly.express as px
fig = px.box(subdf, x="day", y="Slope_normalized", points="all", color='type', hover_data=["Independent","Dependent","Slope","measurement","pullNo"])
# Nastavení osy x jako kategorie
fig.update_layout(
    xaxis=dict(
        type='category'
    )
)
fig.show()
# -



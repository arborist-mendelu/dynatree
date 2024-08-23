#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# %% [markdown]
# # Vyhodnocení statiky
#
# [Online verze](https://jupyter.mendelu.cz/user/dynatree/lab/workspaces/auto-s/tree/tahovky_regrese/static_pull_vyhodnoceni.ipynb) se všemi vstupy a příkazy.
#
# ------
# Technická: Výstup udělat na lokále takto
# ~~~
# mv ~/Downloads/static_pull_vyhodnoceni.ipynb .
# jupyter nbconvert --to html  --no-input --no-prompt static_pull_vyhodnoceni.ipynb
# ~~~
#
# ------
#
# Summary:
#
# * Nebrat data pro náklon lana z TXT souborů RpeAngle(100), ale z Bářiny xls tabulky a pevnou hodnotu pro celý experiment.
# * Regresní koeficienty jsou hezčí, když máme interval 10-90 procent Fmax, ale jenom o málo a z jiných důvodů se víc hodi interval 30-90 procent.
# * Druhé a další zatáhnutí vykazují o pět procent nižší tuhost než první v grafu moment versus náklon, p<0.001. TODO: opravit na jednostrannou hypotézu, ale vyjde to i tak.
#
# Načteme data a zahodíme měření, která nevyšla. To je jedno meření, kdy neměřil siloměr.

# %%
"""
Created on Wed Aug 21 06:21:13 2024

Pracuje s daty z  csv_output/regresions_static.csv, vytvoreneho skriptem 
static_pull.py


@author: marik
"""

# nastaveni pro jupyter
import sys
import os
prefix = "/babice/Mereni_Babice_zpracovani"
if os.path.isdir(prefix):
    # jupyter, pridej cestu ke skriptum
    sys.path.append('/babice/Mereni_Babice_zpracovani/skripty/')
else:
    # lokalni PC, oprav cestu k datum
    prefix = '..'

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lib_dynatree

measurements_with_leaves_true = ["2021-06-29", "2022-08-16"]



# %%
# Načtení a vyčištění dat, report kde selhalo zpracování.
df = pd.read_csv("csv_output/regresions_static.csv", index_col=0)
df["measurement"] = "M0"+ df["measurement"].str[-1]
df = df.drop(['p-value', 'stderr', 'intercept_stderr'], axis=1)
# df = df[~((df["Dependent"].str.contains("Pt")) & (df["measurement"]=="M01"))]

df_ignore = {}
# Ignore undefined values
mask = df.isnull().any(axis=1)
df_ignore["failed"] = df[mask].copy()
df = df[np.logical_not(mask)]

# Ignore the minor axis (the regressions are not great)
mask = df["Dependent"].str.contains("_Min")
df_ignore["Min"] = df.loc[mask,:].copy()
df = df.loc[np.logical_not(mask),:]

# Ignore the major axis (similar to tal value)
mask = df["Dependent"].str.contains("_Maj")
df_ignore["Maj"] = df.loc[mask,:].copy()
df = df.loc[np.logical_not(mask),:]

# Set information about leaves.
df["leaves"] = False
idx = (df["date"].isin(measurements_with_leaves_true))
df.loc[idx,"leaves"] = True

df_save = df.copy()
df_ignore["failed"][["date", "tree", "measurement"]].drop_duplicates()


# %% [markdown]
# ## Liší se data která byla vypočtena jinak stanovenými úhly?
#
# Zjistíme, jestli jeden postup dává jinou kvalitu než jiný. Najdeme rozdíl mezi R^2 jedním a druhým způsobem.
#
# Histogram je vpravo. Málokdy je Rope lépe korelovanný než Measure.
#

# %%

# For lower bound 30% compare data with _Rope and _Measurement. The difference is
# in the determinantion of the angle. With _Rope the angle is from the device
# but it is highly variable and sometime unrealistic. With _Measurement the angle
# is constant. Also report data with highest difference between Rope-like and 
# Measure-like regressions.

df_tail = {}
for bound in [0.1,0.3]:
    df = df_save.copy()
    df = df[df["lower_bound"]==bound]
    # Split _Rope and _Measure
    dfs = {}
    for method in ["Rope", "Measure"]:
        mask = df["Independent"].str.contains(method)
        dfs[method] = df[mask.values]
        dfs[method].loc[:,"Independent"] = dfs[method].loc[:,"Independent"].str.replace(f"_{method}","")
        col_dict = {
            "Slope": f"Slope_{method}",
            "Intercept": f"Intercept_{method}",
            "R^2": f"R^2_{method}",
            }
        dfs[method] = dfs[method].rename(columns=col_dict)
    
    df_both = pd.merge(dfs["Rope"], dfs["Measure"], on=[
        'Independent', 'Dependent', 'pull', 'date', 'tree', 'measurement', 'lower_bound', 'upper_bound', 'leaves'])
    
    df = df_both.copy()
    
    df.loc[:,"R^2_diff"] = df["R^2_Measure"] - df["R^2_Rope"]
    df.loc[:,"R^2_diff_abs"] = np.abs(df.loc[:,"R^2_diff"])
    df = df.sort_values(by="R^2_diff_abs")
    
    fig, ax = plt.subplots()
    df["R^2_diff"].hist(bins=20, range=(-0.5,.5), ax=ax)
    ax.set(yscale='log', title=f'The difference between R^2 for Rope and\n Measurement angle determination, lower bound is {bound}',
           xlabel="R^2_Measure - R^2_Rope"
          )
    plt.tight_layout()

    df_tail[bound] = df.tail(n=50)
# %% [markdown]
# 10 měření s největším rozdílem dat při jiné metodice stanovení úhlu. Samostatně pro dolní mez 10 a 30 procent.

# %%
to_drop = ["lower_bound", "upper_bound", "leaves"] + sum( [[f"Slope_{i}", f"Intercept_{i}", f"R^2_{i}"] for i in ["Rope", "Measure"]],[])
df_tail[0.1].tail(10).drop(to_drop, axis=1)

# %%
df_tail[0.3].tail(10).drop(to_drop, axis=1)


# %% [markdown]
# ## Co se dělo s úhlem tam, kde jsou nevjvětší rozíly v korelačním koeficinetu R^2?

# %%
def bad2image(bad_corr,bound):
    fig, ax = plt.subplots(3,3, figsize = (10,10), sharey=True)
    ax = ax.reshape(-1)
    for i,row in bad_corr.iloc[:9,:].reset_index().iterrows():
        filename = f"{prefix}/data/pulling_tests/{row['date'].replace('-','_')}/BK_{row['tree'][-2:]}_M{row['measurement'][-1]}.TXT"
        df = lib_dynatree.read_data_inclinometers(filename)
        df.plot(y="RopeAngle(100)", title=filename.replace(prefix+"/data/pulling_tests/",""), style='.', ax = ax[i])
    plt.suptitle(f"Lower bound: {bound}")
    plt.tight_layout()
    return fig
# %% [markdown]
# Obrázky, jak vypadá RopeAngle tam, kde je nevětší rozdíl v regresích. Všímej, že někdy vyletí na 90 stupňů. Potom je moment najednou nulový, i když už jsme v pokročilé fázi napínání. Výsledkem je, že v diagramu moment/úhel je kromě relativně velkých momentů i několik nesmyslných nulových, které jsou ale daleko a proto hodně ovlivní lineární regresi.

# %%
for i in [0.1,0.3]:
    bad_corr = df_tail[i][["date", "tree", "measurement"]].drop_duplicates().tail(10)
    bad2image(bad_corr, i)
# %% [markdown]
# ## Jsou korelační koeficienty lepší při odřezávání u 30 nebo 10 procent?
#
# Zajímá mě to, ale důležitější je pochpitelně, co dává větší smysl. 
#
# Podle histogramu to vypadá, že korelční koeficinety jsou o malinko lepší, pokud máme data už od 10 procent.

# %%
df = df_save.copy()
# Pivotování DataFrame
df_wide = df.pivot_table(index=['date', 'tree', 'measurement', 'Independent', 'Dependent', 'pull'],
                         columns='lower_bound', values='R^2').reset_index()
df_wide["rozdil"] = df_wide[0.3] - df_wide[0.1]
df_wide['rozdil'].hist(bins=40)

# %%

# %% [markdown]
# ## Jsou regresní koeficienty M versus blue závislé na přítomnosti listí?

# %%

fig, axs = plt.subplots(2,1,figsize=(14,10), sharex=True)

ax = axs[0]

savedf = df_both.copy()
df = savedf.query("Independent == 'M' and Dependent == 'blue'")
sns.boxplot(
    data=df, 
    x="tree", 
    y="Slope_Measure", 
    hue="leaves",
    ax = ax
    )
ax = axs[1]

sns.boxplot(
    data=df, 
    x="tree", 
    y="Intercept_Measure", 
    hue="leaves",
    ax = ax
    )

[ax.axvline(x+.5,color='gray', lw=0.5) for x in ax.get_xticks()]
ax.legend(loc=2, title="Leaves")
ax.grid(alpha=0.4)
# %% [markdown]
# ## Platí, že u prvního zatáhnutí je větší tuhost?

# %%
df = df_save.copy()
df = df[
    (df["measurement"]=="M01") & 
    (df["lower_bound"]==0.3) & 
    # (df["Dependent"]=="yellow") & 
    (df["Independent"]=="M_Measure")
    ].drop(["lower_bound","upper_bound","measurement"], axis=1)
tests = df[["Independent","Dependent","date","tree"]].drop_duplicates()


# %% [markdown]
# ChatGPT: Mam tabulku se sloupci date, tree, Independent, Dependent, pull, Slope. Chtel bych kazdou hodnotu Slope vydelit cislem, ktere je v tabulce pro stejne hodnoty day, tree, Independent a Dependent a hodnota pull je rovna nule.

# %%
# Nejprve vytáhneme hodnoty Slope pro pull=0
df_zero_pull = df[df['pull'] == 0].copy()

# Přejmenujeme sloupec Slope, aby bylo jasné, že jde o referenční hodnoty
df_zero_pull = df_zero_pull.rename(columns={'Slope': 'Slope_zero_pull', 'Intercept': 'Intercept_zero_pull'})

# Merge původního DataFrame s tím, kde je pull=0, na základě společných sloupců
df_merged = pd.merge(df, df_zero_pull[['date', 'tree', 'Independent', 'Dependent', 'Slope_zero_pull', 'Intercept_zero_pull']],
                     on=['date', 'tree', 'Independent', 'Dependent'], how='left')

# Vydělení hodnoty Slope referenční hodnotou Slope_zero_pull
for i in ['Slope', 'Intercept']:
    df_merged[f'{i}_normalized'] = df_merged[i] / df_merged[f'{i}_zero_pull']


# %%
# Kontrola
df_merged[df_merged["pull"]==0].loc[:,["Slope_normalized", "Intercept_normalized"]].drop_duplicates()

# %%
target = "Slope"
subdf = df_merged[df_merged["pull"]!=0].loc[:,["pull","Intercept_normalized","Slope_normalized","tree"]]
fig, ax = plt.subplots(1,1,figsize=(14,5))
subdf['pull'] = subdf['pull'].astype('category')
sns.swarmplot(data=subdf, x='tree', y=f'{target}_normalized', hue='pull', ax=ax)
ax.grid()

# %% [markdown]
# ChatGPT: Mam tabulku se sloupci tree a Slope_normalized. Tree jsou kategorie. Chci pro kazdou kategorii prumer a smerodatnou odchylku.
#
# ChatGPT: Chci nyni pro kazdou kategorii otestovat hypotezu, ze prumer se nelisi od jedne.

# %%
from scipy.stats import ttest_1samp

result1 = subdf.groupby('tree')['Slope_normalized'].agg(['mean', 'std']).reset_index()

# Funkce pro testování hypotézy
def test_hypothesis(group):
    t_stat, p_value = ttest_1samp(group['Slope_normalized'], 1)
    return pd.Series({'t_stat': t_stat, 'p_value': p_value})

# Aplikace funkce na skupiny dle kategorie 'tree'
result2 = subdf.groupby('tree').apply(test_hypothesis, include_groups=False).reset_index()

result = pd.merge(result1,result2, on=["tree"])
result["H0_rejected"] = result["p_value"]<0.05
result.sort_values(by="H0_rejected")

# %% [markdown]
# Vypadá to, že jenom pro pět stromů je možno zamítnout hypotézu, že v prvním zatáhnutí je stejná směrnice jako v těch dalších. Testovány jsou směrnice v grafu Moment versus náklon blue a yellow. 
#
# Protože máme všechno bezrozměrné, můžeme data pro jednotlivé stromy spojit dohromady a potom to vyjde líp: Nulová hypotéza se může zamítnout, první tah je jiný (tužší) než ty další. Na druhou stranu, průměr se moc neliší od jedničky (ṕokud vezmeme v úvahu i směrodatnou odchylku).

# %%
ttest_1samp(subdf['Slope_normalized'], 1)


# %%
ax = subdf["Slope_normalized"].reset_index(drop=True).plot(style=".")
prumer = subdf["Slope_normalized"].mean()
odchylka = subdf["Slope_normalized"].std()
length = len(subdf["Slope_normalized"])
plt.axhline(prumer, color='red', linestyle='--', label=f'Mean {prumer:.4f}')
plt.fill_between(np.arange(length), prumer - odchylka, prumer + odchylka, color='gray', alpha=0.3, label=f'Standard Deviation {odchylka:.4f}')
ax.grid()
ax.legend()

# %%
prumer, odchylka

# %% [markdown]
# ## Pokus o odstranění odlehlých měření

# %%
data = subdf["Slope_normalized"].values

# Výpočet kvartilů a IQR
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

# Určení hranic pro odlehlé hodnoty
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Odstranění odlehlých hodnot
filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
plt.plot(filtered_data,".")
plt.grid()

# %% [markdown]
# Taky to vyjde na zamítntí nulové hypotézy s p<0.001.

# %%

ttest_1samp(filtered_data, 1)


# %%
filtered_data.mean()

# %%

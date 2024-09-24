"""
Created on Wed Aug 21 06:21:13 2024

Pracuje s daty z anotated_regressions_static.csv, vytvoreneho skriptem 
static_pull.py a static_pull_anotatte_regressions.py


@author: marik
"""

# nastaveni pro jupyter
import sys
import os
import plotly.express as px
import plotly
from plotly.subplots import make_subplots

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

# ['2021-03-22',
#  '2021-06-29',
#  '2021-08-03',
#  '2022-04-05',
#  '2022-08-16',
#  '2023-07-17',
#  '2024-01-16',
#  '2024-04-10',
#  '2024-09-02']

# +
# df = df.query("not (Dependent == 'blue' and tree == 'BK08' and measurement == 'M03' and day == '2021-03-22')")
# df = df.query("not (Dependent == 'Elasto-strain' and tree == 'BK10' and measurement == 'M04' and day == '2022-04-05')")
# df = df.query("not (Dependent == 'Elasto-strain' and tree == 'BK10' and measurement == 'M04' and day == '2022-08-16')")
# df = df.query("not (Dependent == 'Elasto-strain' and tree == 'BK08' and measurement == 'M03' and day == '2024-09-02')")
# df = df.query("not (Dependent == 'blue' and tree == 'BK13' and measurement == 'M01' and day == '2024-09-02' and 'pullNo' == '0')")
# -


def read_data():
    days_with_leaves_true = ["2021-06-29", "2021-08-03", "2022-08-16", "2023-07-17", "2024-09-02"]
    days_after_first_reduction = ['2024-01-16', '2024-04-10', '2024-09-02']
    
    # Načtení a vyčištění dat
    df = pd.read_csv("../outputs/anotated_regressions_static.csv", index_col=0)
    
    df['evaluation'] = df.apply(lambda row: 1 if row['failed'] 
                                else (2 if row['R^2'] < 0.5 else 0), axis=1)
    
    df = df.drop(["p-value","stderr", "intercept_stderr", "upper_cut"], axis=1).reset_index(drop=True)
    df = df[df["tree"] != "JD18"]
    
    df = df.dropna(subset=["Independent","Dependent"], how='all')
    # df[["Independent","Dependent"]].drop_duplicates()
    df = df[df["lower_cut"]==0.3]
    
    # Set information about leaves.
    df.loc[:,"leaves"] = False
    idx = (df["day"].isin(days_with_leaves_true))
    df.loc[idx,"leaves"] = True
    
    # ignore optics
    df = df[df["optics"]==False]
    
    df.loc[:,"reductionNo"] = 0
    idx = (df["day"].isin(days_after_first_reduction))
    df.loc[idx,"reductionNo"] = 1
    idx = (df["type"]=="afterro")
    df.loc[idx,"reductionNo"]=1
    idx = (df["type"]=="afterro2")
    df.loc[idx,"reductionNo"]=2
    
    df = df.drop(["optics","lower_cut"], axis=1).reset_index(drop=True)
    df = df[df["Dependent"].isin(["blue","yellow","Elasto-strain"])]
    
    pairs = df[["Independent","Dependent"]].drop_duplicates()
    
    df["state"] = df["leaves"].astype(str) + ", " +df["reductionNo"].astype(str)
    return df

def main(remove_failed=False, trees=None):
    df = read_data()        
    if trees is None:
        trees = df["tree"].drop_duplicates().values
    df['reason'] = df['reason'].fillna("")
    if remove_failed:
        df = df[~df["failed"]]
    f_ans = {}
    for tree in trees:
        fig = make_subplots(rows=1, cols=3, subplot_titles=("M/blue", "M/yellow", "M_Elasto/Elasto-strain"))
        f = {}
        for I,_ in enumerate(zip(["M","M","M_Elasto"],["blue", "yellow", "Elasto-strain"])):
            i,d = _
            f[I] = px.strip(df[(df["Independent"]==i) & (df["Dependent"]==d) & (df["tree"]==tree)], 
                         x="state", y="Slope", #points="all", 
                         hover_data=['day', 'tree', "measurement", "type", "pullNo", "R^2", "reason", "Dependent"], width=1000, height=500, 
                         color='evaluation',   
                         template =  "plotly_white",
                         # color_discrete_sequence=px.colors.qualitative.Set1,  # Nastavení barevné škály
                         title = f"Slope in {i} versus {d}")
            for trace in f[I]['data']:
                fig.add_trace(trace, row=1, col=1+I)
        fig.update_layout(height=400, width=1200, title_text=f"Tree {tree}")
        fig.update_layout(showlegend=False, template =  "plotly_white",)
        f_ans[tree] = fig
    return f_ans

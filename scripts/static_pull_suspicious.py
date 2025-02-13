#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 12:37:02 2024

@author: marik
"""

import pandas as pd
from dynatree import dynatree, static_pull
import matplotlib.pyplot as plt
import matplotlib
import config
from parallelbar import progress_map

df = pd.read_csv(config.file["outputs/anotated_regressions_static"], index_col=0)
df = df.dropna(subset=["Independent","Dependent"],how='all')
df = df[df["lower_cut"]==0.3]
df = df[df["optics"]==False]
df = df.dropna(how='all', axis=0)
df = df[df['Independent'].isin(['blueMaj', 'yellowMaj', 'Elasto-strain'])]
# df = df[~df['Dependent'].str.contains('Min')]
df = df.sort_values(by="R^2")
df = df[(df['R^2'] < 0.9) | (df['failed']==True)]

print(df.shape)

def plot_row(row):
    if row['Independent'] in ["Pt3","Pt4"]:
        return
    m = static_pull.DynatreeStaticMeasurement(
        day=row['day'], tree=row['tree'], measurement=row['measurement'], 
        measurement_type=row['type'])
    pull = m.pullings[row['pullNo']]
    fig, ax = plt.subplots(3,2,figsize=(12,8))
    data = pull.data
    if row['Independent'] in ['blue', 'blueMaj']:  # consider only Major
        data.plot(y=["Inclino(80)X","Inclino(80)Y"], ax=ax[0,1], style='.')
        data.plot(x="Force(100)", y=["Inclino(80)X","Inclino(80)Y"], ax=ax[2,0], style='.')
        m.data_pulling.plot(y=["Inclino(80)X","Inclino(80)Y"], ax = ax[0,0], style='.')
    
    if row['Independent'] in ['yellow','yellowMaj']: # consider only Major
        data.plot(y=["Inclino(81)X","Inclino(81)Y"], ax=ax[0,1], style='.')
        data.plot(x="Force(100)", y=["Inclino(81)X","Inclino(81)Y"], ax=ax[2,0], style='.')
        m.data_pulling.plot(y=["Inclino(81)X","Inclino(81)Y"], ax = ax[0,0], style='.')
        
    if row['Independent'] == 'Elasto-strain':
        data.plot(y=["Elasto(90)"], ax=ax[0,1], style='.')
        data.plot(x="Force(100)", y=["Elasto(90)"], ax=ax[2,0], style='.')
        m.data_pulling.plot(y=["Elasto(90)"], ax = ax[0,0], style='.')

    data.plot(y=["Force(100)"], ax=ax[1, 1], style='.')
    m.data_pulling.plot(y=["Force(100)"], ax=ax[1, 0], style='.')
    ax[2,1].text(0,0.5,row['reason'], wrap=True)
    ax[2,1].axis('off')

    ax[0, 0].grid()
    ax[0, 1].grid()
    ax[1, 0].grid()
    ax[1, 1].grid()
    ax[2, 0].grid()

    plt.suptitle(f"{pull.measurement_type} {pull.day} {pull.tree} {row['measurement']} pullNo={row['pullNo']} R^2={row['R^2']:.4f}")
    plt.tight_layout()
    return fig,ax
#%%

#%%

def process_one_row(row):
    out = plot_row(row)
    if out is not None:
        fig, ax = out
        filename = f"{row['type']}_{row['day']}_{row['tree']}_{row['measurement']}_{row['pullNo']}_{row['Dependent']}"
        fig.savefig(f"../temp/static_fail_images/{filename}.png")
        plt.close('all')

def main():
    try:
        matplotlib.use('TkAgg') # https://stackoverflow.com/questions/39270988/ice-default-io-error-handler-doing-an-exit-pid-errno-32-when-running
    except:
        matplotlib.use('Agg')
    # for i, row in df.iterrows():
    #     process_one_row(row)
    progress_map(process_one_row, df.to_dict('records'))

if __name__ == "__main__":
    main()
    

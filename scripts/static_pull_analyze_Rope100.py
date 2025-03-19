#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graf smerodatnych odchylek veliciny RopeAngle(100). 
Puvodne zamysleno jako podpurny argument, proc do vypoctu momentu musime 
brat konstantni hodnotu a ne tu namerenou behem napinani.


Created on Mon Sep  9 21:28:09 2024

@author: DYNATREE project, ERC CZ no. LL1909 "Tree Dynamics: Understanding of Mechanical Response to Loading"
"""

from dynatree import static_pull as sp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sp.get_all_measurements(method='all', type='all')
ans_data = {}
cut = 0.3
use_optics = False

for i,row in df.iterrows():
    msg = f"Processing {row['day']} {row['tree']} {row['measurement']}, {row['type']}, optics availability is {row['optics']}"
    # try:
        # get regressions for two cut-out values and merge
    
    data_obj = sp.DynatreeStaticMeasurement(day=row['day'], tree=row['tree'], measurement=row['measurement'], measurement_type=row['type'], optics=use_optics, restricted=(cut,0.9))
    if data_obj.parent.data_pulling is None:
        print(f"There are no data for pulling tests for this case. {data_obj.parent}")
        continue
    
    for i,pull in enumerate(data_obj._get_static_pulling_data(restricted=(0.3,0.9), optics=False)):
        ans = {}
        ans["day"] = row['day']
        ans["tree"] = row['tree']
        ans["measurement"] = row['measurement']
        ans["type"] = row['type']
        ans["pullNo"] = i
        ans_data[(i, row['day'], row['tree'], row['measurement'], row['type'])] = {
            'mean' : pull["RopeAngle(100)"].mean(),
            'std' : pull["RopeAngle(100)"].std()
            }
df_ans = pd.DataFrame(ans_data).T
df_ans['quotient'] = df_ans['std']/df_ans['mean']
print(f"Median of quotients std(Rope)/mean(Rope) is {df_ans.median()}")


df_ans = df_ans.reset_index().rename({'level_0':"pull", 'level_1':'day'}, axis=1)


df_ans['std'].plot(style='.')

fig,ax = plt.subplots(figsize=(10,6))
sns.swarmplot(df_ans, y='std', x='day', size=2)

plt.xticks(rotation=90)
plt.title("Směrodatná odchylka veličiny RopeAngle(100) při natahovací fázi static pulling")
plt.tight_layout()
plt.savefig("../outputs/static_pulling_std_RopeAngle100.pdf")
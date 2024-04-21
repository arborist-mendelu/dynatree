#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 17:37:18 2024

@author: marik
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("damping/damping_results.csv")

fig, ax = plt.subplots(figsize=(15,6))

sns.boxplot(
    data=df, 
    x="tree", 
    y="k", 
    hue="date",
    ax = ax
    )
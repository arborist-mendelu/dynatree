#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 09:07:25 2023

@author: marik
"""

from dynatree.dynatree import DynatreeMeasurement
from time import time

m = DynatreeMeasurement(day="2021-03-22", tree="BK04", measurement="M01")
start = time()
df = m.data_acc5000_single("a01_x")
end = time()
print(end-start)

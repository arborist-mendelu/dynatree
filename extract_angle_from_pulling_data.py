#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 23:38:06 2024

@author: marik
"""
import re

with open('BK01_M01.TXT', 'r') as file:
    lines = [next(file).replace("\t"," ") for _ in range(50)]  # Načítá prvních 50 řádků

match = re.search(r'^a_rope\s+([\d,]+)', "".join(lines), re.MULTILINE)

if match:
    # Nahradí desetinnou čárku tečkou
    number_str = match.group(1).replace(',', '.')
    
    # Převede na float
    number = float(number_str)
    
number
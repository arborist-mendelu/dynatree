#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 07:52:51 2023

Analyzuje pomoci FFT jedno mereni  

@author: marik
"""

from FFT_spectrum import do_fft_for_file, create_fft_image

date = "2021-03-22"
tree = "01"
measurement = "2"
start  = 63

output = do_fft_for_file(
    date=date, tree=tree, measurement=measurement, return_image=True, start=start)

create_fft_image(**output[(date,tree,measurement,('Pt3','Y0'))])

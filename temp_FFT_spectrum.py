#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 07:52:51 2023

Analyzuje pomoci FFT jedno mereni  

@author: marik
"""

from FFT_spectrum import do_fft_for_file

date = "01_Mereni_Babice_22032021_optika_zpracovani"
tree = "01"
measurement = "2"
start  =63


output = do_fft_for_file(date=date, tree=tree, measurement=measurement, return_image=True, start=start)
output.keys()
output['figure']


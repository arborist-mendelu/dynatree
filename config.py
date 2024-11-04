#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 07:03:03 2024

@author: marik
"""

import os
try:
    PREFIX = os.environ["PREFIX_DYNATREE"]
except:
    PREFIX = ""

file = {}

file['static_fail'] = "csv/static_fail.csv"
file['static_checked_OK'] = "csv/static_checked_OK.csv"
file['scale_factors'] = "csv/scale_factors.csv"
file['reset_inclinometers'] = "csv/reset_inclinometers.csv"
file['synchronization_finetune_inclinometers_fix'] = "csv/synchronization_finetune_inclinometers_fix.csv"
file['angles_measured'] = "csv/angles_measured.csv"

file["outputs/regressions_static"] = "../outputs/regressions_static.csv"
file['outputs/anotated_regressions_static'] = "../outputs/anotated_regressions_static.csv"

file["tsv_dirs"] = "csv/tsv_dirs.csv"

file["FFT_release"] = "csv/FFT_release.csv"
file["FFT_failed"] = "csv/FFT_failed.csv"

file['outputs/FFT_csv_tukey'] = "../outputs/FFT_csv_tukey.csv"
file['FFT_manual_peaks'] = "csv/FFT_manual_peaks.csv"
file['solara_FFT'] = "csv/solara_FFT.csv"
file['FFT_comments'] = "csv/FFT_comments.csv"
file['FFT_failed'] = "csv/FFT_failed.csv"

for key in file.keys():
    file[key] = PREFIX + file[key]



file['logfile'] = '/tmp/dynatree.log'
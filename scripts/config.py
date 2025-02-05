#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 07:03:03 2024

@author: marik
"""

import os

PREFIX = os.environ.get("PREFIX_DYNATREE", "")
datapath = os.environ.get("DYNATREE_DATAPATH", "../data")

file = {}

file['static_fail'] = "csv/static_fail.csv"
file['static_checked_OK'] = "csv/static_checked_OK.csv"
file['scale_factors'] = "csv/scale_factors.csv"
file['reset_inclinometers'] = "csv/reset_inclinometers.csv"
file['synchronization_finetune_inclinometers_fix'] = "csv/synchronization_finetune_inclinometers_fix.csv"
file['angles_measured'] = "csv/angles_measured.csv"
file['oscillation_times_remarks'] = "csv/oscillation_times_remarks.csv"
file["solara_FFT"] = "csv/solara_FFT.csv"
file['static_manual_limits'] = "csv/static_manual_limits.csv"


file["outputs/regressions_static"] = "../outputs/regressions_static.csv"
file['outputs/anotated_regressions_static'] = "../outputs/anotated_regressions_static.csv"
file['outputs/peak_width'] = "../outputs/peak_width.csv"
file['outputs/damping_factor'] = "../outputs/damping_factor.csv"


file["tsv_dirs"] = "csv/tsv_dirs.csv"

file["FFT_release"] = "csv/FFT_release.csv"
file["FFT_failed"] = "csv/FFT_failed.csv"

file['outputs/FFT_csv_tukey'] = "../outputs/FFT_csv_tukey.csv"
file['FFT_manual_peaks'] = "csv/FFT_manual_peaks.csv"
file['solara_FFT'] = "csv/solara_FFT.csv"
file['FFT_comments'] = "csv/FFT_comments.csv"
file['FFT_failed'] = "csv/FFT_failed.csv"

file['outputs/FFT_acc_knock'] = "../outputs/FFT_acc_knock.csv"
file['cachedir'] = "../outputs/cache"
file['cachedir_large'] = "../outputs/fft_images_knocks"

file["trans_vse.csv"] = f"{datapath}/puda/trans_vse.csv"
file["penetrologgers"] = f"{datapath}/puda/Puda/penetrologger*"

for key in file.keys():
    file[key] = PREFIX + file[key]

file['logfile'] = '/tmp/dynatree.log'

summer_dates = ["2021-06-29", "2022-08-16", "2021-08-03", "2023-07-17", "2024-09-02"]
damping_threshold = 0.15

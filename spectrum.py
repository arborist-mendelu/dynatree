#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 07:52:51 2023

@author: marik
"""

import pandas as pd
from lib_dynatree import read_data
from lib_dynatree import directory2date
from lib_dynatree import filename2tree_and_measurement_numbers
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.fft import fft, fftfreq
import os

df_remarks = pd.read_csv("csv/oscillation_times_remarks.csv")

# measurement_day = MEASUREMENT_DAY
# path = PATH

# %% Create subdirectories

def do_fft_for_file(
        path="../", 
        measurement_day="01_Mereni_Babice_22032021_optika_zpracovani",
        csvdir="csv",
        tree="01",
        tree_measurement="2",
        start = 0,
        end = np.inf,
        column_fft=("Pt3","Y0"),
        create_image=True,
        color="C1",
        return_image=False,
        ):
    if np.isnan(start) or np.isnan(end) or (start==end):
        print("Nejsou zadany meze pro signal")
        return None
    df = read_data(f"{path}{measurement_day}/{csvdir}/BK{tree}_M0{tree_measurement}.csv")
    
    signal_ = df[column_fft].dropna().loc[start:end]
    if np.isnan(signal_).any():
        signal_ = signal_.interpolate()
    if signal_.shape[0] == 0:
        print("Nebyl nalezen signal signal")
        return None
    time = signal_.index  # grab time
    time = time - time[0] # restart time from zero
    signal = signal_.values # grab values
    fs = 100
    time_fft = np.arange(time[0],time[-1],1/fs) # timeline for resampling
    f = interpolate.interp1d(time, signal, fill_value="extrapolate")  
    signal_fft = f(time_fft) # resample
    signal_fft = signal_fft - np.nanmean(signal_fft) # mean value to zero
    
    N = time_fft.shape[0]  # get the number of points
    
    if create_image:
        fig,axs = plt.subplots(2,1)  
        ax = axs[0]  # plot the signal
        ax.plot(time_fft,signal_fft, color=color)
        ax.set(xlabel="Time/s", ylabel=column_fft[0]+", "+column_fft[1])
    
    yf = fft(signal_fft)  # preform FFT analysis
    xf_r = fftfreq(N, 1/fs)[:N//2]
    yf_r = 2.0/N * np.abs(yf[0:N//2])
    peak_index = np.argmax(yf_r[2:])+2  # find the peak, exclude the start
    peak_position = xf_r[peak_index]
    delta_f = np.diff(xf_r).mean()
    output = {'peak position': peak_position, 'delta f': delta_f}

    if create_image:
        ax = axs[1]
        ax.plot(xf_r, yf_r,".", color=color)
        ax.plot(xf_r[peak_index],yf_r[peak_index],"o", color='red')
        t = ax.text(xf_r[peak_index]+0.1,yf_r[peak_index],f"{round(xf_r[peak_index],3)}±{np.round(delta_f,3)}",horizontalalignment='left')
        t.set_bbox(dict(facecolor='yellow', alpha=0.5))
        ax.set(xlim=(0,3))
        ax.grid()
        ax.set(ylabel="FFT", xlabel="Freq./Hz", yscale='log', ylim=(0.001,None))
    
        plt.suptitle(directory2date(measurement_day)+f", BK{tree}_M0{tree_measurement}")
        plt.tight_layout()
        fig.savefig(f"{path}{measurement_day}/png_fft/BK{tree}_M0{tree_measurement}.png")
        if return_image:
            output['figure']=fig
        else:
            plt.close()
    return output


# a = do_fft_for_file(measurement_day="01_Mereni_Babice_29062021_optika_zpracovani", 
                    # tree=24, tree_measurement="4", start=128.5, end=165.46, column_fft=("Pt3","Y0"),return_image=True)
# plt.show(a['figure'])
    
def do_fft_for_day(
        measurement_day="01_Mereni_Babice_22032021_optika_zpracovani",
        path="./",
        color="C0"
        ):
    for d in ["png_fft"]:
        try:
           os.makedirs(f"{path}{measurement_day}/{d}")
        except FileExistsError:
           # directory already exists
           pass
    
    fft_data = {}
    
    csvdir="csv"
    files = os.listdir(f"{path}{measurement_day}/{csvdir}/")
    files.sort()
    
    for file in files[:]:
        print(file, end="")    

        tree,tree_measurement = filename2tree_and_measurement_numbers(file)
        bounds_for_fft = df_remarks[(df_remarks["tree"]==f"BK{tree}") & (df_remarks["measurement"]==f"M0{tree_measurement}") & (df_remarks["date"]==directory2date(measurement_day))]
        if bounds_for_fft['probe'].isnull().values.any():
            column_fft = ("Pt3","Y0") 
        else:
            column_fft = (bounds_for_fft['probe'].iat[0],"Y0") 
            
        start = bounds_for_fft['start'].iat[0]
        end = bounds_for_fft['end'].iat[0]
        print(f"{column_fft} from {start} to {end}: ", end="")        
        output_fft = do_fft_for_file(
            start=start, 
            end=end,
            column_fft=column_fft,
            create_image=True,
            measurement_day=measurement_day,
            tree=tree,
            tree_measurement=tree_measurement,
            color=color,
            )
        if output_fft is not None:
            peak_position = output_fft['peak position']
            delta_f = output_fft['delta f']        
        
            print (np.round(peak_position,3), "±",np.round(delta_f,3))
            fft_data[file.replace(".csv","")] = [peak_position, delta_f]
        else:
            print("output is None")

    df_output = pd.DataFrame(fft_data).T
    df_output.columns = ["Freq","Delta freq"]
    df_output.to_excel(f"fft_data_{measurement_day}.xlsx")


for MEASUREMENT_DAY, COLOR in [
        ["01_Mereni_Babice_22032021_optika_zpracovani", "C0"],
        ["01_Mereni_Babice_29062021_optika_zpracovani", "C1"],
        ["01_Mereni_Babice_05042022_optika_zpracovani", "C0"],
        ["01_Mereni_Babice_16082022_optika_zpracovani", "C1"],
        ]:
    print(MEASUREMENT_DAY)
    do_fft_for_day(measurement_day=MEASUREMENT_DAY, color=COLOR, path="../")

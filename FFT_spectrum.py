#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 07:52:51 2023

Analyzuje pomoci FFT bud jedno mereni nebo celý den nebo všechny dny. 
Je možné načíst jako knihovnu nebo spustit jako program. Pokud je spuštěn jako
program, dělá FFT analýzu pro všechna měření ve všech dnech. 

@author: marik
"""

import pandas as pd
from lib_dynatree import read_data_selected, date2dirname
from lib_dynatree import get_all_measurements

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

df_remarks = pd.read_csv("csv/oscillation_times_remarks.csv", index_col=[0,1,2])

# measurement_day = MEASUREMENT_DAY
# path = PATH

def interp(df, new_index):
    """Return a new DataFrame with all columns values interpolated
    to the new_index values."""
    df_out = pd.DataFrame(index=new_index, columns=df.columns)
    df_out.index.name = df.index.name

    for colname in df.columns:
        df_ = df[colname].dropna()
        try:
            df_out[colname] = np.interp(new_index, df_.index, df_.values, right=np.nan)
        except:
            df_out[colname] = np.nan
    return df_out

def extend_dataframe_with_zeros(data, tail=2):
    """
    Input: dataframe with time in the index
    Output: copy of the dataframe with zero signal at the initial and final 
            part
    
    The mean value is subtracted from the data and zero sequences are
    added on the initial and final part of the data.
    The index is supposed to be time in regular steps. The length of the 
    added sequence is in the variable tail.
    """
    data_pad = data.copy()
    data_pad = data_pad.dropna()
    data_pad = data_pad - data_pad.mean()
    dt = data_pad.index[1]-data_pad.index[0]
    time_start = np.arange(data_pad.index[0]-tail, data_pad.index[0], dt)
    time_end = np.arange(data_pad.index[-1]+dt, data_pad.index[-1]+tail, dt)
    time_extended = np.concatenate([time_start,time_end])
    df_extended = pd.DataFrame(index = time_extended)
    df_extended[data.columns] = 0
    data_pad = pd.concat([data_pad, df_extended])
    data_pad = data_pad.sort_index()
    return data_pad

def extend_series_with_zeros(data, tail=2):
    """
    Wrapper for extending series by zeros converting to dataframe
    and using extend_dataframe_with_zeros function
    """
    df = pd.DataFrame(data)
    df = extend_dataframe_with_zeros(df, tail=tail)
    return df[df.columns[0]]

def load_data_for_FFT(
        file="../01_Mereni_Babice_05042022_optika_zpracovani/csv/BK04_M02.csv", 
        start=100, 
        end=120, 
        dt=0.01, 
        probes = ["Time"]
            +[f"Pt{i}" for i in [0,1,3,4]]
            +[f"BL{i}" for i in range(44,68)],
        filter_cols=True
        ):
    """
    Loads data for FFT. The time will be in the index. Only selected probes are included.
    """
    data = read_data_selected(file, probes = probes)
    data = data.set_index("Time")
    idx = pd.isna(data.index)
    data = data[~idx]
    col = data.columns
    if filter_cols:
        col = [i for i in col if 
               ("P"==i[0][0] and "Y0"==i[1])
               or ("B"==i[0][0] and "Pt0AY"==i[1])
               ]
    data = data.loc[start:end, col]
    data = interp(data, np.arange(data.index[0],data.index[-1],dt))
    data = data - data.iloc[0,:]
    return data

# %% Create subdirectories

def do_fft_for_file(
        path="../", 
        date="01_Mereni_Babice_22032021_optika_zpracovani",
        csvdir="csv",
        tree="01",
        df=None,
        measurement="2",
        start = 0,
        end = np.inf,
        column_fft=("Pt3","Y0"),
        create_image=True,
        color="C1",
        return_image=True,
        save_image=False
        ):
    """
    FFT analyza zvoleneho probu v csv souboru    

    Parameters
    ----------
    path : TYPE, optional
        Cesta k datům. The default is "../".
    measurement_day : TYPE, optional
        Adresář s daty. The default is "01_Mereni_Babice_22032021_optika_zpracovani".
    csvdir : TYPE, optional
        Adresář s csv soubory. The default is "csv".
    tree : TYPE, optional
        Dvouciferné číslo stromu. The default is "01".
    tree_measurement : TYPE, optional
        Jendociferné číslo měření. The default is "2".
    df : You can pass the dataframe as a parameter. If df is None, reads from the disc.
    start : TYPE, optional
        Začátek časového intervalu. The default is 0.
    end : TYPE, optional
        Konec časového intervalu. The default is np.inf.
    column_fft : TYPE, optional
        Sloupec pro FFT. The default is ("Pt3","Y0").
    create_image : TYPE, optional
        The default is True.
    color : TYPE, optional
        Barva pro odlišení s listy a bez listů. The default is "C1".
    return_image : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    """
    if np.isnan(start) or np.isnan(end) or (start==end):
        print("Nejsou zadany meze pro signal")
        return None
    if df is None:
        df = load_data_for_FFT(f"{path}{date}/{csvdir}/BK{tree}_M0{measurement}.csv",start=start, end=end)
    output = {}
    for c in df.columns:
        output[(date,tree,measurement,c)] = do_fft_for_one_column(df,c,color=color,
                              return_image=return_image,
                              save_image=save_image,
                              create_image=create_image, 
                              tree=tree, 
                              measurement=measurement, 
                              date=date,
                              path=path
        )
    return output

def do_fft_for_one_column(df,
                          col,
                          dt=0.01,
                          create_image=True, 
                          save_image=False, 
                          return_image=True,
                          color="C0",
                          tree=None,
                          measurement=None,
                          date=None,
                          path=None,
                          preprocessing = None
                          ):
    signal_fft = df[col].dropna().copy()
    if preprocessing is not None:
        signal_fft = preprocessing(signal_fft)
    time_fft = signal_fft.index.values  # grab time
    signal_fft = signal_fft.values
    if signal_fft.shape[0] == 0:
        print("Nebyl nalezen signal")
        return None
    time_fft = time_fft - time_fft[0]
    signal_fft = signal_fft - np.nanmean(signal_fft) # mean value to zero
    N = time_fft.shape[0]  # get the number of points

    yf = fft(signal_fft)  # preform FFT analysis
    xf_r = fftfreq(N, dt)[:N//2]
    yf_r = 2.0/N * np.abs(yf[0:N//2])
    peak_index = np.argmax(yf_r[2:])+2  # find the peak, exclude the start
    peak_position = xf_r[peak_index]
    delta_f = np.diff(xf_r).mean()
    output = {
        'peak_position': peak_position, 
        'delta_f': delta_f, 
        'xf_r': xf_r, 
        'yf_r': yf_r,
        'peak_index': peak_index, 
        'signal_fft': signal_fft, 
        'time_fft': time_fft
        }
    return output


def create_fft_image(
        time_fft=None, 
        signal_fft=None, 
        color="C01", 
        col="", 
        xf_r=None, 
        yf_r=None, 
        peak_index=None, 
        delta_f=None, 
        date=None, 
        tree=None, 
        measurement=None, 
        path="", 
        peak_position=None, 
        only_fft = False,
        ymin = 0.001
        ):
    if only_fft:
        fig,ax = plt.subplots(1,1)  
    else:
        fig,axs = plt.subplots(2,1)  
        ax = axs[0]  # plot the signal
        ax.plot(time_fft,signal_fft-signal_fft[0], ".", color=color, ms=2)
        ax.set(xlabel="Time/s",ylabel=col)
        ax = axs[1]
    ax.plot(xf_r, yf_r,".", color=color)
    ax.plot(xf_r[peak_index],yf_r[peak_index],"o", color='red')
    t = ax.text(xf_r[peak_index]+0.1,yf_r[peak_index],f"{round(xf_r[peak_index],3)}±{np.round(delta_f,3)}",horizontalalignment='left')
    t.set_bbox(dict(facecolor='yellow', alpha=0.5))
    ax.set(xlim=(0,3))
    ax.grid()
    ax.set(ylabel="FFT", xlabel="Freq./Hz", yscale='log', ylim=(ymin,None))

    # plt.suptitle(directory2date(date)+f", BK{tree}_M0{measurement}")
    plt.tight_layout()
    return fig

def main():
    df = get_all_measurements()
    output_data = {}
    probes = [f"Pt{i}" for i in [3,4]] +[
        f"BL{i}" for i in range(44,68)]
    for date,tree,measurement in df.values:
        bounds_for_fft = df_remarks.loc[[(date,f"BK{tree}",f"M0{measurement}")],:]
        start = bounds_for_fft[['start']].iat[0,0]
        end = bounds_for_fft[['end']].iat[0,0]
        if pd.isna(start):
            print("Start not defined")
            continue
        if start < .1:
            print("Start is not set")
            continue
        print(f"{date} BK{tree} M0{measurement} from {start} to {end}, ", end="")        
        data = load_data_for_FFT(
            file=f"../{date2dirname(date)}/csv/BK{tree}_M0{measurement}.csv",
            start=start,end=end)
        print(", ",round(data.index[-1]-data.index[0],1)," sec.")
        for probe in probes:
            if probe[0] == "P":
                probe = (probe,"Y0")
            else:
                probe = (probe,"Pt0AY")            
            try:
                output = do_fft_for_one_column(
                    data, 
                    probe
                    )
            except:
                print("Something failed in do_fft_for_one_columne")
                output = None
            if output is None:
                print(f"Probe {probe} failed")
                output_data[(date,tree,measurement,probe)
                      ] = [np.nan]*3
                continue
            print(date,probe, round(output['peak_position'],6), "±",np.round(output['delta_f'],6))
            length = output['time_fft'][-1] - output['time_fft'][0]
            output_data[(date,tree,measurement,probe)
                  ]  = [output['peak_position'], output['delta_f'], length]
    return output_data

if __name__ == "__main__":
    output_data = main()
    df = pd.DataFrame(output_data).T
    df.columns=['freq','err','length']
    df.index.names = ["date","tree","measurement","probe"]
    df.to_csv("results/fft.csv")
    # date = "2021-03-22"
    # tree = "01"
    # measurement = "2"
    # probe = 'Pt3'
    # if probe[0] == "P":
    #     probe = (probe,"Y0")
    # else:
    #     probe = (probe,"Pt0AY")            
    # data = load_data_for_FFT(
    #         file=f"../{date2dirname(date)}/csv/BK{tree}_M0{measurement}.csv",start=63.4, end=70.54)
    # output = do_fft_for_one_column(
    #     data, 
    #     probe
    #     )
    # output
    # fig = create_fft_image(**output)

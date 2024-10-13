#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 15:07:36 2023

Vykreslí v jednom grafu optiku, inklinometry, ...
Umožní posoudit, jestli došlo k bezproblémové synchronizaci a jestli je 
dobře vybrán interval pro release.


@author: marik
"""

import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib
import argparse
from lib_dynatree import read_data, read_data_selected, find_release_time_optics, find_release_time_interval
from lib_dynatree import find_finetune_synchro, DynatreeMeasurement
from dynatree_util import read_data_inclinometers
from static_pull import DynatreeStaticPulling
import pathlib
from tqdm import tqdm

def plot_one_measurement(
        date="2021-03-22",
        path="../data",
        tree="01",
        measurement="2",
        df_remarks=None,
        return_figure=True,
        save_figure=False,
        xlim=(None,None),
        df_extra=None,
        df=None,
        figsize=(10,7), 
        plot_fixes=True, 
        plot_Pt4=False,
        major_minor=False,
        release_detail=False,
        img_path="../temp"
        ):
    """
    Vykreslí tři obrázky. 
    V horním je pohyb Pt3 a pootm Pt3 s odečtením posunu bodů na zemi.
    V prostředním obrázku jsou inklinoměry.
    V dolním obrázku jsou síla, přeškálovaná výchylka z prvního obrázku tak, 
    aby se dala porovnat se silou a v samostatné soustavě souřadnic 
    data z eleastometru.
    
    major_minor: pokud je True, kresli se major a minor. Jinak inclino(80)X a inclino(80)Y
    release_detail: pokud je True, kresli se detail okolo vypusteni
    
    df: do not read csv but use this one DataFrame instead

    Returns
    -------
    fig : 

    """
      
    m = DynatreeMeasurement(date, tree, measurement)
    if df_remarks is None:
        df_remarks = pd.read_csv("csv/oscillation_times_remarks.csv")
    
    # accept both M02 and 2 as a measurement number
    measurement = measurement[-1]
    # accept both BK04 and 04 as a tree number
    tree = tree[-2:]
    date = date.replace("-","_")
    
    if df is None:
        df = m.data_optics_pt34
    else:
        # print("Skipping csv reading: "+f"{path}{measurement_day}/csv/BK{tree}_M0{measurement}.csv")
        pass
    if df_extra is None:
        df_extra = m.data_optics_extra

    draw_from,draw_to = xlim
    if draw_from == None:
        draw_from = 0
    if draw_to == None:
        draw_to = df.index.max()

    bounds_for_fft = df_remarks[
        (df_remarks["tree"] == f"BK{tree}") & 
        (df_remarks["measurement"] == f"M0{measurement}") & 
        (df_remarks["date"] == date.replace("_","-"))
            ]
    if len(bounds_for_fft)>0:
        lower_bound = bounds_for_fft["start"].values[0]
        upper_bound = bounds_for_fft["end"].values[0] 
        description = bounds_for_fft['remark'].values[0]
    else:
        lower_bound = 0
        upper_bound = 0
        description = "Static measurement"
    fix_target = 3
    plot_coordiante = "Y"
    fixes = [
        i for i in df_extra.columns if f"{fix_target}_fixed_by" 
        in i[0] and plot_coordiante in i[1]]

    fig, axes = plt.subplots(3,1,figsize=figsize,sharex=True)
    plt.suptitle(
        f"{date.replace('_optika_zpracovani','')} - BK{tree} M0{measurement}")

    # Plot probes, region of interest for oscillation
    ax = axes[0]
    if plot_Pt4:
        tempdata = df.loc[draw_from:draw_to,[('Pt3', f'{plot_coordiante}0'),('Pt4', f'{plot_coordiante}0')]].copy()
        tempdata = tempdata - tempdata.iloc[0,:]
        tempdata.plot(ax=ax)
        ax.set(title=f"Distance of Pt3 and Pt4 from the initial position")    
    else:
        if plot_fixes:
            df_extra.loc[draw_from:draw_to,fixes].plot(ax=ax)
            ax.set(title=f"Pt{fix_target} and fixes based on points on ground")    
        else:
            ax.set(title=f"Pt{fix_target}")    
        df.loc[draw_from:draw_to,(f'Pt{fix_target}', f'{plot_coordiante}0')].plot(ax=ax)

    ax.legend(title="",loc=2)
    ax.grid()
    t = ax.text(
        0, 0, 
        description,
        ha='left', 
        va='bottom',
        transform=ax.transAxes,
        color="r",
        backgroundcolor="white",
        wrap=True)
    t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor='white'))

    if upper_bound == np.inf:
        upper_bound = df["Time"].max()
    ax.axvspan(lower_bound, upper_bound, alpha=0.5, color="gray")
        
    # plot inclinometers
    ax = axes[1]    
    list_inclino = ["Inclino(80)X","Inclino(80)Y","Inclino(81)X","Inclino(81)Y"]
    delta_time = find_finetune_synchro(date, tree,measurement) 
    # print("delta time",delta_time)

    # načte synchronizovaná data a přesampluje na stejné časy jako v optice
    if measurement == "1":
        release_time_optics = 0
    else:
        release_time_optics = find_release_time_optics(df)
    # print ("release time",release_time_optics, "measurement",measurement)
    
    df_pulling_tests = read_data_inclinometers(
        m, 
        release=release_time_optics, 
        delta_time=delta_time
        )    
    for inclino in list_inclino:
        bounds = find_finetune_synchro(date, tree,measurement, inclino) 
        if bounds is None or np.isnan(bounds).any():
            continue
        start,end = bounds
        inclino_mean = df_pulling_tests.loc[start:end,inclino].mean()
        df_pulling_tests[inclino] = df_pulling_tests[inclino] - inclino_mean
    if major_minor:
        axes_major = DynatreeMeasurement(date, tree, measurement).identify_major_minor
        list_major_minor = ["blueMaj", "blueMin", "yellowMaj","yellowMin"]
        df_major_minor = DynatreeStaticPulling(
            df_pulling_tests, ini_forces=False, ini_regress=False
            ).data
        for i in list_major_minor:
            df_pulling_tests[i] = df_pulling_tests[
                axes_major[i.replace("blue","Inclino(80)").replace("yellow","Inclino(81)")]]
        # df_major_minor = process_inclinometers_major_minor(df_pulling_tests)
        df_major_minor.loc[draw_from:min([draw_to,df_major_minor.index[-1]]),list_major_minor].plot(ax=ax, style=".")
        ax.legend(list_major_minor, title="", loc=3)
    else:
        df_pulling_tests.loc[draw_from:draw_to,list_inclino].plot(ax=ax, style=".")
        ax.legend(list_inclino, title="", loc=3)
    ax.grid()
    ax.set(title="Inclinometers")
        
    # plot force and strain
    ax = axes[2]

    f = df[(f'Pt{fix_target}', f'{plot_coordiante}0')].copy().dropna()
    f = f-f.iloc[0]
    fmax = np.nanmax(f.values)
    fmin = np.nanmin(f.values)
    if np.abs(fmax) > np.abs(fmin):
        f = f / fmax
    else:
        f = f / fmin
    f = f * df_pulling_tests.loc[:,"Force(100)"].max()
    f.plot(ax = ax)
    
    ax.plot(df_pulling_tests.index,df_pulling_tests.loc[:,"Force(100)"],".", label='Force')
    if xlim[0] is not None and xlim[1] is not None and xlim[1]-xlim[0]<15:
        maj_pos = ticker.MultipleLocator(1)   # major ticks 
    else:
        maj_pos = ticker.MultipleLocator(10)   # major ticks 
    min_pos = ticker.MultipleLocator(1)    # minor ticks 
    ax.xaxis.set(major_locator=maj_pos, minor_locator=min_pos)
    ax.grid(which='major')
    ax.grid(which='minor', lw=1)
    ax.xaxis.set(major_locator=maj_pos, minor_locator=min_pos)
    lines1, labels1 = ax.get_legend_handles_labels()
    ax.legend().remove()    
    ax = ax.twinx()
    df_pulling_tests.loc[:,"Elasto(90)"].plot(ax=ax,color="C2", style=".")
    ax.set(title=f"Force, Elasto, scaled (Pt{fix_target}, {plot_coordiante}0)")
    lines2, labels2 = ax.get_legend_handles_labels()
    ax.legend(lines1 + lines2, [f"scaled (Pt{fix_target}, {plot_coordiante}0)","Force","Elasto"], loc=3)
    ax.grid(color="C2")
    ax.tick_params(axis='y', labelcolor="C2")     

    for ax in axes:
        ax.axvline(x = release_time_optics, color='k', linestyle="dashed", zorder=0)

    tmin, tmax = find_release_time_interval(df_extra, date, tree, measurement)
    
    for ax in axes:
        # Nasledujici radek omezi graf na okamzik okolo vypusteni
        if release_detail:
            ax.set(xlim=(tmin-(tmax-tmin), max(tmax+2*(tmax-tmin),release_time_optics)))
        ax.axvspan(tmin,tmax, alpha=.5, color="yellow")
        # pre_release_data[file.replace(".csv","")] = delta_df.mean()
        ax.set(xlim=xlim, ylim=(None, None))        
    axes[2].set(ylim=(0,None))
    fig.tight_layout()
    if save_figure:
        pathlib.Path(f"{img_path}/optics_with_inclino/").mkdir(parents=True, exist_ok=True)
        fig.savefig(
            f"{img_path}/optics_with_inclino/{date}_BK{tree}_M0{measurement}.pdf")
    if return_figure:
        return fig
    else:
        plt.close(fig)

def plot_one_day(date="2021-03-22", path="../data", release_detail=False):
    
    files =  glob.glob(f"../data/parquet/{date.replace('-','_')}/BK??_M??.parquet")
    files.sort()
    pbar = tqdm(total=len(files))
    for file in files:
        filename = file.split("/")[-1].replace(".parquet","")
        # print(filename,", ",end="", flush=True)
        tree, measurement = filename.split("_")
        pbar.set_description(f"{tree} {measurement}")
        plot_one_measurement(
            date=date, 
            path=path, 
            tree=tree[-2:], 
            measurement=measurement[-1], 
            save_figure=True, 
            return_figure=False, 
            major_minor=True, 
            release_detail=release_detail
            )
        pbar.update(1)

    pbar.close()
    print(f"Konec zpracování pro {date}")
    
def main():
    # answer = input("The file will create png files with Pt3 movement, inclinometers, force and elastometer.\nOlder data (if any) will be replaced.\nConfirm y or yes to continue.")
    # if answer.upper() in ["Y", "YES"]:
    #     pass
    # else:
    #     print("File processing skipped.")
    #     return None    
    
    try:
        matplotlib.use('TkAgg') # https://stackoverflow.com/questions/39270988/ice-default-io-error-handler-doing-an-exit-pid-errno-32-when-running
    except:
        matplotlib.use('Agg')
    parser = argparse.ArgumentParser(description="Explore quality of synchronization between optics and extenso/inclinometers.")
    # Přidání argumentu s volbou True/False
    parser.add_argument('--release_detail', action='store_true', help='Kreslí detail okolo vypuštění (True/False)')
    # Parsování argumentů
    args = parser.parse_args()
    for i in [
            "2021-03-22", 
            "2021-06-29", 
            "2022-04-05", 
            "2022-08-16"
                    ]:
        print(i)
        print("=====================================================")
        plot_one_day(date=i, release_detail=args.release_detail)
        # break

if __name__ == "__main__":
    main()


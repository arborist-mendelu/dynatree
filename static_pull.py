# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:25:19 2024

@author: marik
"""

DIRECTORY = "../data"
import pandas as pd
# import glob
# import os
# from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from scipy.interpolate import interp1d
from lib_dynatree import read_data_inclinometers, read_data, timeit
from functools import lru_cache
# import seaborn as sns

def split_df_static_pulling(
        df_ori, 
        intervals=None,
        ):
    """
    Analyzes data in static tests, with three pulls. Inputs the dataframe, 
    outputs the dictionary. 
    
    output['times'] contains the time intervals of increasing pulling force. 
    output['df_interpolated'] return interpolated force values. 
    
    Initial estimate of subintervals can be provided in a file
    csv/intervals_split_M01.csv. If not, the initial guess is created 
    automatically. 
    
    Method description:
        * drop nan values of force
        * interpolate and smooth out
        * find intervals where function is decreasing from top down
        * find a maximum and then the minimum which preceeds this maximum    
    
    """
    df= df_ori.copy()[["Force(100)"]].dropna()
    
    # Interpolace a vyhlazeni
    new_index = np.arange(df.index[0],df.index[-1],0.1)
    window_length = 100
    polyorder = 3
    newdf = df[["Force(100)"]].dropna()
    interpolation_function = interp1d(newdf.index, newdf["Force(100)"], kind='linear')
    df_interpolated = pd.DataFrame(interpolation_function(new_index), index=new_index, columns=["Force(100)"])
    df_interpolated['Force_smoothed'] = savgol_filter(df_interpolated['Force(100)'], window_length=window_length, polyorder=polyorder)

    steps_down=None
    if intervals is None:
        # Divide domain into interval where force is large and small
        maximum = df_interpolated["Force(100)"].max()
        df_interpolated["Force_step"] = (df_interpolated["Force(100)"]>0.5*maximum).astype(int) 

        # identify jumps down
        diff_d1 = df_interpolated["Force_step"].diff()
        steps_down = list(diff_d1.index[diff_d1<0])
        intervals = zip([0]+steps_down[:-1],steps_down)

    time = []
    for start,end in intervals:
        df_subset = df.loc[start:end,"Force(100)"]
        maximum_idx = np.argmax(df_subset)
        t = {'maximum': df_subset.index[maximum_idx]}

        df_subset = df.loc[start:t['maximum'],"Force(100)"]
        idxmin = df_subset.idxmin()
        t['minimum'] = idxmin

        time = time + [t]
    return {'times':time, 'df_interpolated': df_interpolated}

def find_intervals_to_split_measurements_from_csv(date, tree, csv="csv/intervals_split_M01.csv"):
    """
    Tries to find initial gues for separation of pulls in M1 measurement
    from csv file. If the measturement is not included (most cases), return None. 
    In this case the splitting is done automatically.
    """
    df = pd.read_csv(csv, index_col=[], dtype={"tree":str}, sep=";")
    select = df[(df["date"]==date) & (df["tree"]==tree)]
    if select.shape[0] == 0:
        return None
    elif select.shape[0] > 1:
        print (f"Warning, multiple pairs of date-tree in file {csv}")
        select = select.iat[0,:]
    return np.array([
        int(i) for i in (
            select["intervals"]
                .values[0]
                .replace("[","")
                .replace("]","")
                .split(",")
                )
        ]).reshape(-1,2)


def process_inclinometers_major_minor(df_, height_of_anchorage=None, rope_angle=None, 
                          height_of_pt=None):
    """
    The input is dataframe loaded by pd.read_csv from pull_tests directory.
    
    * Converts Inclino(80) and Inclino(81) to blue and yellow respectively.
    * Evaluates the total angle of inclination from X and Y part
    * Adds the columns with Major and Minor axis.
    """
    df = pd.DataFrame(index=df_.index, columns=["blue","yellow"])
    df[["blueX","blueY","yellowX","yellowY"]] = df_[["Inclino(80)X","Inclino(80)Y", "Inclino(81)X","Inclino(81)Y",]]
    for inclino in ["blue","yellow"]:
        df.loc[:,[inclino]] = arctand(
            np.sqrt((tand(df[f"{inclino}X"]))**2 + (tand(df[f"{inclino}Y"]))**2 )
            )
        # najde maximum bez ohledu na znamenko
        maxima = df[[f"{inclino}X",f"{inclino}Y"]].abs().max()
        # vytvori sloupce blue_Maj, blue_Min, yellow_Maj,  yellow_Min....hlavni a vedlejsi osa
        if maxima[f"{inclino}X"]>maxima[f"{inclino}Y"]:
            df.loc[:,[f"{inclino}_Maj"]] = df[f"{inclino}X"]
            df.loc[:,[f"{inclino}_Min"]] = df[f"{inclino}Y"]
        else:
            df.loc[:,[f"{inclino}_Maj"]] = df[f"{inclino}Y"]
            df.loc[:,[f"{inclino}_Min"]] = df[f"{inclino}X"]
        # Najde pozici, kde je extremalni hodnota - nejkladnejsi nebo nejzapornejsi
        idx = df[f"{inclino}_Maj"].abs().idxmax()
        # promenna bude jednicka pokus je extremalni hodnota kladna a minus
        # jednicka, pokud je extremalni hodnota zaporna
        if pd.isna(idx):
            znamenko = 1
        else:
            znamenko = np.sign(df[f"{inclino}_Maj"][idx])
        # V zavisosti na znamenku se neudela nic nebo zmeni znamenko ve sloupcich
        # blueM, blueV, yellowM, yellowV
        for axis in ["_Maj", "_Min"]:
            df.loc[:,[f"{inclino}{axis}"]] = znamenko * df[f"{inclino}{axis}"]    
    return df

def process_forces(
        df_, 
        height_of_anchorage=None, 
        rope_angle=None,
        height_of_pt=None
        ):
    """
    Input is a dataframe with Force(100) column
    """
    df = pd.DataFrame(index=df_.index)
    # evaluate the horizontal and vertical component
    if rope_angle is None:
        # If rope angle is not given, use the data from the table
        rope_angle = df_['RopeAngle(100)']
    # evaluate horizontal and vertical force components and moment
    # obrat s values je potreba, protoze df_ ma MultiIndex
    df.loc[:,['F_horizontal']] = (df_['Force(100)'] * np.cos(np.deg2rad(rope_angle))).values
    df.loc[:,['F_vertical']] = (df_['Force(100)'] * np.sin(np.deg2rad(rope_angle))).values
    df.loc[:,['M']] = df['F_horizontal'] * height_of_anchorage
    df.loc[:,['M_Pt']] = df['F_horizontal'] * ( height_of_anchorage - height_of_pt )

    return df
    
def tand(angle):
    """
    Evaluates tangens of the angle. The angli is in degrees.
    """
    return np.tan(np.deg2rad(angle))
def arctand(value):
    """
    Evaluates arctan. Return the angle in degrees.
    """
    return np.rad2deg(np.arctan(value))    

def get_regressions(df, independent="Time"):
    regrese = {}
    dependent = [_ for _ in df.columns if _ !=independent]
    for i in dependent:
        reg = linregress(df[independent].astype(float),df[i].astype(float))
        regrese[i] = [i, reg.slope, reg.intercept, reg.rvalue**2, reg.pvalue, reg.stderr, reg.intercept_stderr]
    df = pd.DataFrame(regrese, index=["Name", "Slope", "Intercept", "R^2", "p-value", "stderr", "intercept_stderr"], columns=dependent).T
    return df

#%%

df_pt_notes = pd.read_csv("csv/PT_notes_with_pt.csv", sep=",")
df_pt_notes.index = df_pt_notes["tree"]

def get_static_pulling_data(day, tree, measurement, directory=DIRECTORY, skip_optics=False):
    """
    Uniform method to find the data. The data ara obtained from pulling tests
    for M01 and from parquet files for the other measurements.
    
    The data from M01 measurement are from the device output. 
    
    If skip_optics is true, the other measurements are handles in the same
    way as M01. 
    
    If skip_optics is False, the data for M02 and higher are from 
    parquet_add_inclino.py library. These data are synchronized with optics and
    recaluculated to the same time index as optics.
    
    """
    measurement = measurement[-1]
    tree = tree[-2:]
    if skip_optics or measurement == "1":
        df = read_data_inclinometers(f"{directory}/pulling_tests/{day.replace('-','_')}/BK_{tree}_M{measurement}.TXT")
        out = split_df_static_pulling(df, intervals = find_intervals_to_split_measurements_from_csv(day, tree))
        times = out['times']
    else:
        df = read_data(f"{directory}/parquet/{day.replace('-','_')}/BK{tree}_M0{measurement}_pulling.parquet")
        times = [{"minimum":0, "maximum": df["Force(100)"].idxmax().values[0]}]
        df = df.drop([i for i in df.columns if i[1]!='nan'], axis=1)
        df.columns = [i[0] for i in df.columns]
    df["Elasto-strain"] = df["Elasto(90)"]/200000
    return {'times': times, 'dataframe': df}

def get_computed_data(day,tree,measurement, out=None, skip_optics=False):
    """
    Gets the data from process_inclinometers_major_minor and from
    process_forces functions.
    
    out is the output of get_static_pulling_data(day, tree, measurement)
    """
    if out is None:
        out = get_static_pulling_data(day, tree, measurement,skip_optics=skip_optics)
    
    ans = {}
    
    ans['inclinometers'] = process_inclinometers_major_minor(out['dataframe'])
    
    ans['forces_from_rope_angle'] = process_forces(
        out['dataframe'], 
        height_of_anchorage=df_pt_notes.at[int(tree),'height_of_anchorage'], 
        height_of_pt=df_pt_notes.at[int(tree),'height_of_pt']
        )

    ans['forces_from_pt_data'] = process_forces(
        out['dataframe'], 
        height_of_anchorage=df_pt_notes.at[int(tree),'height_of_anchorage'], 
        height_of_pt=df_pt_notes.at[int(tree),'height_of_pt'],
        rope_angle=df_pt_notes.at[int(tree),'angle_of_anchorage']
        )

    for n,s in zip(['forces_from_pt_data', 'forces_from_rope_angle'],["PT","Rope"]):
        ans[n].columns = [f"{i}_{s}" for i in ans[n]]
        
    answer = pd.concat([ans[i] for i in ans.keys()], axis=1)
    return answer

def get_interval_of_interest(df, maximal_fraction=0.9, minimal_fraction=0.1):
    """
    The input is the dataframe with maximal value at the end. Return indices
    for values between given fraction of maximal value. 
    """
    maximum = df.iat[-1,0]
    mask = df>maximal_fraction*maximum
    upper = mask.idxmax().iloc[0]
    subdf = df.loc[:upper,:]
    mask = subdf<minimal_fraction*maximum
    lower = mask.iloc[::-1,0].idxmax()
    # lowerindex = mask.shape[0] - mask.iloc[:,0].values[::-1].argmax()
    # lower = mask.index[lowerindex]
    # lower = mask[mask == True].dropna().index.max() 
    return lower, upper
    
#%%

@timeit
def proces_data(day, tree, measurement, skip_optics=False):
    """
    skip_optics False means ifgnore parquet files and read original TXT file.
    """
    # read data, either M02, 3, 4, etc or three pulls in M1
    # interpolate the missing data if necessary
    out = get_static_pulling_data(day, tree, measurement, skip_optics=skip_optics)
    out['dataframe']=out['dataframe'].interpolate(method='index')
    dataframe = out['dataframe']
    tree = tree[-2:]
    measurement = measurement[-1]
    ans = get_computed_data(day, tree, measurement,out)    
    return {'times': out['times'], 'dataframe': pd.concat([dataframe, ans], axis=1)}

@timeit    
def nakresli(day, tree, measurement, skip_optics=False):
    ans = proces_data(day, tree, measurement, skip_optics=skip_optics)
    dataframe = ans['dataframe']
    
    fig, ax = plt.subplots()
    dataframe["Force(100)"].plot(ax=ax)
    ax.set(title=f"Static {day} {tree} {measurement}", ylabel="Force")
    
    figs = []
    for i,_ in enumerate(ans['times']):
        subdf = dataframe.loc[_['minimum']:_['maximum'],["Force(100)"]]
        subdf.plot(ax=ax, legend=False)
    
        # Find limits for given interval of forces
        lower, upper = get_interval_of_interest(subdf, maximal_fraction=0.9)        
        subdf[lower:upper].plot(ax=ax, linewidth=4, legend=False)
        ax.legend(["Síla","Náběh síly","Rozmezí 10 až 90\nprocent max."])

        f,a = plt.subplots(2,2, 
                           figsize=(12,9)
                           )
        a = a.reshape(-1)
        df = dataframe.loc[lower:upper,:]
        
        df.loc[:,["Force(100)"]].plot(ax=a[0], legend=False, xlabel="Time", ylabel="Force", style='.')
        # df columns: ['Force(100)', 'Elasto(90)', 'Elasto-strain', 'Inclino(80)X',
        # 'Inclino(80)Y', 'Inclino(81)X', 'Inclino(81)Y', 'RopeAngle(100)',
        # 'blue', 'yellow', 'blueX', 'blueY', 'yellowX', 'yellowY', 
        # 'blue_Maj', 'blue_Min', 'yellow_Maj', 'yellow_Min',
        # 'F_horizontal_Rope', 'F_vertical_Rope',
        # 'M_Rope', 'M_Pt_Rope', 'F_horizontal_PT', 'F_vertical_PT', 'M_PT',
        # 'M_Pt_PT']
        colors = ["blue","yellow"]
 
        df.loc[:,["blue_Maj", "blue_Min", "yellow_Maj", "yellow_Min"]
               ].plot(ax=a[2], xlabel="Time", style='.')
        a[2].grid()
        a[2].set(ylabel="Angle")
        a[2].legend(title="Inclinometer_axis")
        
        df.plot(x="Force(100)", y=colors, ax=a[1], ylabel="Angle", xlabel="Force", style='.')
        a[1].legend(title="Inclinometers")


        a[3].plot(df["M_PT"], df[colors], '.')
        a[3].set(
                    xlabel = "Momentum from PT", 
                    ylabel = "Angle",
                    )
        a[3].legend(colors)
        
        for _ in [a[0],a[1],a[3]]:
            _.set(ylim=(0,None))
            _.grid()
        f.suptitle(f"Detail, pull {i}")        
        plt.tight_layout()
        figs = figs + [f]
    return [fig] + figs

def main():
    day,tree,measurement = "2021-03-22", "BK01", "M02"
    tree=tree[-2:]
    measurement = measurement[-1]
    nakresli(day, tree, measurement)
    data = proces_data(day, tree, measurement)
    pull_value=0
    subdf = data['dataframe'].loc[data['times'][pull_value]['minimum']:data['times'][pull_value]['maximum'],:]
    lower, upper = get_interval_of_interest(subdf)        
    subdf = subdf.loc[lower:upper,["M_PT","blue","yellow"]]
    get_regressions(subdf, "M_PT")
    
if __name__ == "__main__":
    main()


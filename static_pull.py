# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 13:25:19 2024

Inputs the pulling data. Raw pullling data for M1, preprocessed pulling data 
for M2 and higher (parquet_add_inclino.py script).

Used in solara_major_minor_momentum.py  vit visualizations.

Used to get regressions between variables in the phase of static pulling.

@author: marik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.interpolate import interp1d
from lib_dynatree import read_data_inclinometers, read_data, timeit
# from functools import lru_cache
import lib_dynatree
import re

DIRECTORY = "../data"

def get_all_measurements(cesta=DIRECTORY):
    return lib_dynatree.get_all_measurements(cesta, suffix="TXT", directory="pulling_tests")

def available_measurements(df, day, tree):
    select_rows = (df["day"]==day) & (df["tree"]==tree)
    values = df[select_rows]["measurement"].values
    return list(values)

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
    from csv file. If the measurement is not included (most cases), return None. 
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


def process_inclinometers_major_minor(df_):
    """
    The input is dataframe loaded by pd.read_csv from pull_tests directory.
    
    * Converts Inclino(80) and Inclino(81) to blue and yellow respectively.
    * Evaluates the total angle of inclination from X and Y part
    * Adds the columns with Major and Minor axis.
    
    Returns new dataframe with the corresponding columns
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

def read_tree_configuration():
    file_path = "../data/Popis_Babice_VSE_13082024.xlsx"
    sheet_name = "Prehledova tabulka_zakludaje"
    
    # Načtení dat s vynecháním druhého řádku a nastavením sloupce D jako index
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        skiprows=[1],  # Vynechání druhého řádku
        index_col=0,   # Nastavení čtvrtého sloupce (D) jako index
        nrows=13,       # Načtení 13 řádků s daty
        usecols="D,G,H,I,K,M",  # Načtení pouze sloupců D, G, H, K, L
    )
    
    df.columns=["angle_of_anchorage", "distance_of_anchorage",
             "height_of_anchorage", "height_of_pt",
             "height_of_elastometer"]
    
    return df
# df_pt_notes = pd.read_csv("csv/PT_notes_with_pt.csv", sep=",")
# df_pt_notes.index = df_pt_notes["tree"]
df_pt_notes = read_tree_configuration()

def process_forces(
        df_,
        height_of_anchorage=None,
        rope_angle=None,
        height_of_pt=None, 
        height_of_elastometer=None
        ):
    """
    Input is a dataframe with Force(100) column. Evaluates horizontal and vertical 
    component of the force and moments of force
    """
    df = pd.DataFrame(index=df_.index)
    # evaluate the horizontal and vertical component
    if rope_angle is None:
        # If rope angle is not given, use the data from the table
        rope_angle = df_['RopeAngle(100)']
    # evaluate horizontal and vertical force components and moment
    # obrat s values je potreba, protoze df_ ma MultiIndex
    # shorter names
    df.loc[:,['F_horizontal']] = (df_['Force(100)'] * np.cos(np.deg2rad(rope_angle))).values
    df.loc[:,['F_vertical']] = (df_['Force(100)'] * np.sin(np.deg2rad(rope_angle))).values
    df.loc[:,['M']] = df['F_horizontal'] * height_of_anchorage
    df.loc[:,['M_Pt']] = df['F_horizontal'] * ( height_of_anchorage - height_of_pt )
    df.loc[:,['M_Elasto']] = df['F_horizontal'] * ( 
                height_of_anchorage - height_of_elastometer )
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
        height_of_pt=df_pt_notes.at[int(tree),'height_of_pt'],
        height_of_elastometer=df_pt_notes.at[int(tree),'height_of_elastometer'],
        )

    ans['forces_from_measurements'] = process_forces(
        out['dataframe'],
        height_of_anchorage=df_pt_notes.at[int(tree),'height_of_anchorage'],
        height_of_pt=df_pt_notes.at[int(tree),'height_of_pt'],
        rope_angle=df_pt_notes.at[int(tree),'angle_of_anchorage'],
        height_of_elastometer=df_pt_notes.at[int(tree),'height_of_elastometer'],
        )

    for n,s in zip(['forces_from_measurements', 'forces_from_rope_angle'],["Measure","Rope"]):
        ans[n].columns = [f"{i}_{s}" for i in ans[n]]

    answer = pd.concat([ans[i] for i in ans.keys()], axis=1)
    return answer

def get_interval_of_interest(df, maximal_fraction=0.9, minimal_fraction=0.1):
    """
    The input is the dataframe with maximal value at the end. Return indices
    for values between given fraction of maximal value. 
    
    Used to focus on the interval of interest when processing static pulling data.
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

# @timeit
def process_data(day, tree, measurement, skip_optics=False):
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

# @timeit
def nakresli(day, tree, measurement, skip_optics=False):
    """
    Plot the data as in the Solara app.
    
    `skip_optics` means that the data from optics are not considered. 
    Thus no synchrnonization with optical data and no reset of inclinometers.
    Probabably should be kept `False`.
    """
    ans = process_data(day, tree, measurement, skip_optics=skip_optics)
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


        a[3].plot(df["M_Measure"], df[colors], '.')
        a[3].set(
                    xlabel = "Momentum from Measured rope angle",
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

def main_nakresli():
    day,tree,measurement = "2021-03-22", "BK01", "M02"
    tree=tree[-2:]
    measurement = measurement[-1]
    nakresli(day, tree, measurement)
    data = process_data(day, tree, measurement)
    pull_value=0
    subdf = data['dataframe'].loc[data['times'][pull_value]['minimum']:data['times'][pull_value]['maximum'],:]
    lower, upper = get_interval_of_interest(subdf)
    subdf = subdf.loc[lower:upper,["M_Measure","blue","yellow"]]
    get_regressions(subdf, "M_Measure")

def get_regressions_for_one_measurement(day, tree, measurement):
    data = process_data(day, tree, measurement)
    reg_df = {}
    for pull_value,times in enumerate(data['times']):
        subdf = data['dataframe'].loc[times['minimum']:times['maximum'],:]
        lower, upper = get_interval_of_interest(subdf)
        subdf = subdf.loc[lower:upper,:]
        # df columns: ['Force(100)', 'Elasto(90)', 'Elasto-strain', 'Inclino(80)X',
        # 'Inclino(80)Y', 'Inclino(81)X', 'Inclino(81)Y', 'RopeAngle(100)',
        # 'blue', 'yellow', 'blueX', 'blueY', 'yellowX', 'yellowY',
        # 'blue_Maj', 'blue_Min', 'yellow_Maj', 'yellow_Min',
        # 'F_horizontal_Rope', 'F_vertical_Rope',
        # 'M_Rope', 'M_Pt_Rope', 'F_horizontal_PT', 'F_vertical_PT', 'M_PT',
        # 'M_Pt_PT']        
        reg_df[pull_value] = get_regressions(subdf,
            [
            ["M_PT","blue", "yellow", "Elasto-strain", "Force(100)", "F_horizontal_PT"],
            ["M_Pt_PT","blue", "yellow", "Elasto-strain", "Force(100)", "F_horizontal_PT" ],
            ["Elasto-strain","blue","yellow","M_PT","Force(100)","F_horizontal_PT"],
            ]
            )
        reg_df[pull_value].loc[:,"pull"] = pull_value
    reg_df_sum = pd.concat(list(reg_df.values()))
    reg_df_sum.loc[:,["date","tree","measurement"]] = [day, tree, measurement]
    return reg_df_sum

def get_regressions(df, collist):
    """
    Return regression in dataframe. If collist is not list but column name, this
    column is independent variable and other columns are dependenet variable. 
    
    If colist is list, it is assumed that it is a list of lists. In each sublist, 
    the regression of the first item to the oter ones is evaluated.
    """
    df = df.astype(float)
    if not isinstance(collist, list):
        return get_regressions_for_one_column(df, collist)
    data = [get_regressions_for_one_column(df.loc[:,i], i[0]) for i in collist]
    return pd.concat(data)

def get_regressions_for_one_column(df, independent):
    regrese = {}
    dependent = [_ for _ in df.columns if _ !=independent]
    for i in dependent:
        reg = linregress(df[independent],df[i])
        regrese[i] = [independent, i, reg.slope, reg.intercept, reg.rvalue**2, reg.pvalue, reg.stderr, reg.intercept_stderr]
    df = pd.DataFrame(regrese, index=["Independent", "Dependent", "Slope", "Intercept", "R^2", "p-value", "stderr", "intercept_stderr"], columns=dependent).T
    return df

def main():
    df = get_all_measurements()
    all_data = {}
    for i,row in df.iterrows():
        day = row['day']
        tree = row['tree']
        measurement = row['measurement']
        print (f"Processing {day} {tree} {measurement}")
        try:
            all_data[i] = get_regressions_for_one_measurement(day, tree, measurement)
        except:
            print(f"Failed. Day,tree,measurement : {day},{tree},{measurement}")
    df_all_data = pd.concat(all_data).reset_index(drop=True)
    return df_all_data

if __name__ == "__main__":
    # main_output = main()
    # main_output.to_csv("csv_output/regresions_static.csv")
    main_nakresli()

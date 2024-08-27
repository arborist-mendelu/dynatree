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

import os
import glob

import multi_handlers_logger as mhl

DIRECTORY = "../data"

def get_all_measurements_pulling(cesta=DIRECTORY, suffix='parquet', directory='parquet_pulling'):
    """
    Gets all static measurements. Makes use of data from pulling experiments.
    """
    files = glob.glob(cesta+f"/{directory}/*/*.{suffix}") 
    files = [i.replace(cesta+f"/{directory}/","").replace(f".{suffix}","") for i in files]
    s = pd.Series(files)
    split = s.str.split('/', expand=True)
    info = split.iloc[:,1].str.split('_', expand=True)
    ans = {
        'date': split.iloc[:,0].str.replace("_","-"),
        'tree': info.iloc[:,1],
        'measurement': info.iloc[:,2],
        'type': info.iloc[:,0],
        }
    df = pd.DataFrame(ans).sort_values(by = ["date","tree","measurement"])
    df = df.reset_index(drop=True)
    return df

def get_all_measurements_optics(cesta=DIRECTORY, suffix='parquet', directory='parquet'):
    """
    Gets all measurements with optics.
    """
    files = glob.glob(cesta+f"/{directory}/*/*.{suffix}") 
    files = [i.replace(cesta+f"/{directory}/","").replace(f".{suffix}","") for i in files]
    df = pd.DataFrame(files, columns=["full_path"])
    df[["date","rest"]] = df['full_path'].str.split("/", expand=True)
    df["date"] = df["date"].str.replace("_","-")
    df["is_pulling"] = df["rest"].str.contains("pulling")
    mask = df["is_pulling"]
    dfA = df[np.logical_not(mask)].copy()
    dfB = df[mask].copy()
    dfA[["tree","measurement"]] = dfA["rest"].str.split("_", expand=True)
    dfB[["tree","measurement","pulling"]] = dfB["rest"].str.split("_", expand=True)
    dfA = dfA.loc[:,["date","tree","measurement"]].sort_values(by = ["date","tree","measurement"]).reset_index(drop=True)
    dfB = dfB.loc[:,["date","tree","measurement"]].sort_values(by = ["date","tree","measurement"]).reset_index(drop=True)
    if not dfA.equals(dfB):
        print("Některé měření nemá synchronizaci optika versus tahovky.")
    dfA["type"] = "normal"  # optika je vzdy normal
    return dfA

def get_all_measurements(method='optics', type='normal', *args, **kwargs):
    """
    Find all measurements. If the type is optics, we look for the files from optics only. 
    
    For other types we look for both optics and pulling files and merge informations.
    
    If type is 'all' we return all date. Otherwise only the selected type 
    is returned.
    """
    if method == 'optics':
        return get_all_measurements_optics(*args, **kwargs)
    
    df_o = get_all_measurements_optics()
    df_o["optics"] = True
    df_p = get_all_measurements_pulling()
    df = pd.merge(df_p, df_o, 
                on=['date', 'tree', 'measurement', "type"], 
                how='left')
    if type != 'all':
        df = df[df["type"]==type]  # jenom vybrany typ mereni
    # df = df[df["tree"].str.contains("BK")]  # neuvazuji jedli
    df["optics"] = df["optics"] == True
    df["day"] = df["date"]
    return df

# df = get_all_measurements(method='all', type=)

# df = df[df["measurement"] != "M01"]
# df = df[df["optics"].isnull()]
def available_measurements(df, day, tree, measurement_type):
    select_rows = (df["date"]==day) & (df["tree"]==tree)  & (df["type"]==measurement_type)
    values = df[select_rows]["measurement"].values
    return list(values)

@timeit
def process_data(data_obj, skip_optics=False):
    """
    skip_optics=False means ignore parquet files where optics and pulling is synced 
    and read the original file. 
    
    Toto také zahodí případnou informaci, na jakém intervalu je potřeba vynulovat
    inklinometry.
    """
    if (not skip_optics) & (not data_obj.is_optics_available):
        print(f"Optics not available for {data_obj}, forcing skip_optics=True")
        skip_optics = True
    day, tree, measurement = data_obj.day, data_obj.tree, data_obj.measurement
    # read data, either M02, 3, 4, etc or three pulls in M1
    # interpolate the missing data if necessary
    out = get_static_pulling_data(data_obj, skip_optics=skip_optics)
    # remove nan in index if necessary before interpolation
    idxna = out['dataframe'].index.isna()
    out['dataframe'].iloc[~idxna,:]=out['dataframe'].iloc[~idxna,:].interpolate(method='index')
    dataframe = out['dataframe']
    tree = tree[-2:]
    measurement = measurement[-1]
    if measurement != '1' and not skip_optics:
        pt_3_4 = lib_dynatree.read_data_selected(f"../data/parquet/{day.replace('-','_')}/BK{tree}_M0{measurement}.parquet")
        pt_3_4 = pt_3_4.loc[:,[("Pt3","Y0"),("Pt4","Y0")]]
        pt_3_4 = pt_3_4 - pt_3_4.iloc[0,:]
        pt_3_4.columns = ["Pt3","Pt4"]
    else:
        pt_3_4 = pd.DataFrame(index=dataframe.index,data={"Pt3": np.nan, "Pt4":np.nan})
    ans = get_computed_data(data_obj,out)
    
    df = pd.concat([
        dataframe, 
        ans,
        pt_3_4
        ], axis=1)
    return {'times': out['times'], 'dataframe': df}


@timeit
def nakresli(data_object, skip_optics=False):
    """
    Plot the data as in the Solara app.
    
    `skip_optics` means that the data from optics are not considered. 
    Thus no synchrnonization with optical data and no reset of inclinometers.
    Probabably should be kept `False`.
    """
    day, tree, measurement = data_object.day, data_object.tree, data_object.measurement
    measurement_type = data_object.measurement_type
    ans = process_data(data_object, skip_optics=skip_optics)
    dataframe = ans['dataframe']

    fig, ax = plt.subplots()
    dataframe["Force(100)"].plot(ax=ax)
    ax.set(title=f"Static {day} {tree} {measurement} {measurement_type}", ylabel="Force")

    figs = []
    for i,_ in enumerate(ans['times']):
        subdf = dataframe.loc[_['minimum']:_['maximum'],["Force(100)"]]
        subdf.plot(ax=ax, legend=False)

        # Find limits for given interval of forces
        lower, upper = get_interval_of_interest(subdf, maximal_fraction=0.9)
        subdf[lower:upper].plot(ax=ax, linewidth=4, legend=False)

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
        # 'M_Rope', 'M_Pt_Rope', 'M_Elasto_Rope'
        # 'F_horizontal_Measure', 'F_vertical_Measure',
        # 'M_Measure', 'M_Pt_Measure', 'M_Elasto_Measure']
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
    day,tree,measurement, mt  = "2022-04-05", "BK16", "M02", "normal"
    day,tree,measurement, mt = "2023-07-17", "BK01", "M01", "afterro"
    tree=tree[-2:]
    measurement = measurement[-1]
    data_object = lib_dynatree.DynatreeMeasurement(day, tree, measurement)
    nakresli(data_object)
    data = process_data(data_object)
    pull_value=0
    subdf = data['dataframe'].loc[data['times'][pull_value]['minimum']:data['times'][pull_value]['maximum'],:]
    lower, upper = get_interval_of_interest(subdf)
    subdf = subdf.loc[lower:upper,["M_Measure","blue","yellow"]]
    get_regressions(subdf, "M_Measure",info = f"{day} BK{tree} M0{measurement}")

def get_regressions_for_one_measurement(data_obj, minimal_fraction=0.3, maximal_fraction=0.9, skip_optics=False):
    """
    Get regressions for one measurment. 
    
    Minimal fraction if the bound (in percent of Fmax) where to cut out the
    initial phase. The default is 0.3, i.e. 30%.
    """
    if (not skip_optics) & (not data_obj.is_optics_available):
        print(f"Optics not available for {data_obj}, forcing skip_optics=True")
        skip_optics = True
    data = process_data(data_obj, skip_optics=skip_optics)
    day, tree, measurement = data_obj.day, data_obj.tree, data_obj.measurement
    reg_df = {}
    for pull_value,times in enumerate(data['times']):
        subdf = data['dataframe'].loc[times['minimum']:times['maximum'],:]
        lower, upper = get_interval_of_interest(subdf, minimal_fraction=minimal_fraction)
        subdf = subdf.loc[lower:upper,:]
        # df columns: ['Force(100)', 'Elasto(90)', 'Elasto-strain', 'Inclino(80)X',
        # 'Inclino(80)Y', 'Inclino(81)X', 'Inclino(81)Y', 'RopeAngle(100)',
        # 'blue', 'yellow', 'blueX', 'blueY', 'yellowX', 'yellowY',
        # 'blue_Maj', 'blue_Min', 'yellow_Maj', 'yellow_Min',
        # 'F_horizontal_Rope', 'F_vertical_Rope',
        # 'M_Rope', 'M_Pt_Rope']        
        if measurement[-1] != "1":
            pt_reg = [
                ["M_Pt_Rope","Pt3", "Pt4"],
                ["M_Pt_Measure","Pt3", "Pt4"],
                ]
        else:
            pt_reg = []

        reg_df[pull_value] = get_regressions(subdf,
            [
            ["M_Rope",   "blue", "yellow", "blue_Maj", "blue_Min", "yellow_Maj", "yellow_Min"],
            ["M_Measure","blue", "yellow", "blue_Maj", "blue_Min", "yellow_Maj", "yellow_Min"],
            ["M_Elasto_Rope", "Elasto-strain"],
            ["M_Elasto_Measure", "Elasto-strain"],
            ]+pt_reg,
            info=f"{day} BK{tree} M0{measurement}"
            )
        reg_df[pull_value].loc[:,"pull"] = pull_value
    reg_df_sum = pd.concat(list(reg_df.values()))
    reg_df_sum.loc[:,["date","tree","measurement","lower_bound","upper_bound"]] = [day, tree, measurement,minimal_fraction,maximal_fraction]
    return reg_df_sum

def get_regressions(df, collist, info=""):
    """
    Return regression in dataframe. If collist is not list but column name, this
    column is independent variable and other columns are dependenet variable. 
    
    If colist is list, it is assumed that it is a list of lists. In each sublist, 
    the regression of the first item to the oter ones is evaluated.
    """
    if not isinstance(collist, list):
        return get_regressions_for_one_column(df, collist, info=info)
    data = [get_regressions_for_one_column(df.loc[:,i], i[0], info=info) for i in collist]
    return pd.concat(data)

def get_regressions_for_one_column(df, independent, info=""):
    regrese = {}
    dependent = [_ for _ in df.columns if _ !=independent]
    for i in dependent:
        # remove nan valules, if any
        cleandf = df.loc[:,[independent,i]].dropna()
        # do regresions without nan
        try:
            reg = linregress(cleandf[independent],cleandf[i])
            regrese[i] = [independent, i, reg.slope, reg.intercept, reg.rvalue**2, reg.pvalue, reg.stderr, reg.intercept_stderr]
        except:
            pass
            # logger.error(f"Linear regression failed for {independent} versus {i}. Info: {info}")
    ans_df = pd.DataFrame(regrese, index=["Independent", "Dependent", "Slope", "Intercept", "R^2", "p-value", "stderr", "intercept_stderr"], columns=dependent).T
    return ans_df

def main():
    logger = mhl.setup_logger(prefix="static_pull_")
    logger.info("========== INITIALIZATION OF static-pull.py  ============")
    df = get_all_measurements(method='all', type='all')
    # drop missing optics
    # df = df[~((df["day"]=="2022-04-05")&(df["tree"]=="BK21")&(df["measurement"]=="M5"))]
    all_data = {}
    for i,row in df.iterrows():
        day = row['day']
        tree = row['tree']
        measurement = row['measurement']
        optics = row['optics']
        measurement_type = row['type']
        msg = f"Processing {day} {tree} {measurement}, {measurement_type}, optics availability is {optics}"
        logger.info(msg)
        try:
            # get regressions for two cut-out values and merge
            ans_10 = get_regressions_for_one_measurement(day, tree, measurement,minimal_fraction=0.1)
            ans_30 = get_regressions_for_one_measurement(day, tree, measurement,minimal_fraction=0.3)
            ans = pd.concat([ans_10,ans_30])            
            all_data[i] = ans
        except:
            msg = f"Failed. Day,tree,measurement : {day},{tree},{measurement}"
            logger.error(msg)
    df_all_data = pd.concat(all_data).reset_index(drop=True)
    return df_all_data

if __name__ == "__main__":
    # day, tree, measurement, mt = "2022-08-16", "BK11", "M02", "normal"
    # day,tree,measurement, mt = "2023-07-17", "BK01", "M01", "afterro"

    # # day, tree, measurement = "2021-06-29", "BK08", "M04"
    # data_obj = lib_dynatree.DynatreeMeasurement(day, tree, measurement, measurement_type=mt)
    # ans = process_data(data_obj, skip_optics=True)

    # ans_10 = get_regressions_for_one_measurement(data_obj,minimal_fraction=0.1, skip_optics=False)
    # ans_30 = get_regressions_for_one_measurement(day, tree, measurement,minimal_fraction=0.3)
    # ans = pd.concat([ans_10,ans_30])
    # main_nakresli()
    
    # These two lines are for production code to do final analysis
    main_output = main()
    main_output.to_csv("csv_output/regresions_static.csv")

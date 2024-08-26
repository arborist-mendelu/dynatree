#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:35:00 2024

@author: marik
"""

import lib_dynatree
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


class DynatreeStaticMeasurment(lib_dynatree.DynatreeMeasurement):

    def __init__(self, *args, lower_cut=30, upper_cut=90, pullNo=0, **kwargs):
        super().__init__(**kwargs)
        self.lower_cut = lower_cut
        self.pullNo = pullNo
        self.intervalsOI = self.find_intervals_to_split_measurements_from_csv()

    def find_intervals_to_split_measurements_from_csv(self, csv="csv/intervals_split_M01.csv"):
        """
        Tries to find initial gues for separation of pulls in M1 measurement
        from csv file. If the measurement is not included (most cases), return None. 
        In this case the splitting is done automatically.
        """
        if self.measurement != "M01":
            return None
        df = pd.read_csv(csv, index_col=[], dtype={"tree": str}, sep=";")
        select = df[(df["date"] == self.date) & (
            df["tree"] == self.tree[-2:]) & (df["type"] == self.measurement_type)]
        if select.shape[0] == 0:
            return None
        elif select.shape[0] > 1:
            print(f"Warning, multiple pairs of date-tree in file {csv}")
            select = select.iat[0, :]
        return np.array([
            int(i) for i in (
                select["intervals"]
                .values[0]
                .replace("[", "")
                .replace("]", "")
                .split(",")
            )
        ]).reshape(-1, 2)

    def get_static_pulling_data(self, directory="../data", skip_optics=False):
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
        if skip_optics or self.measurement == "M01":
            if self.measurement == "M01":
                intervals_ini = self.find_intervals_to_split_measurements_from_csv()
                out = self.split_df_static_pulling(
                    intervals = intervals_ini)
                times = out['times']
            else:
                times = [{"minimum":0, "maximum": self.data_pulling["Force(100)"].idxmax()}]                
        else:
            df = read_data(f"{directory}/parquet/{day.replace('-','_')}/{pref}{tree}_M0{measurement}_pulling.parquet")
            times = [{"minimum":0, "maximum": df["Force(100)"].idxmax().iloc[0]}]
            df = df.drop([i for i in df.columns if i[1]!='nan'], axis=1)
            df.columns = [i[0] for i in df.columns]
        df["Elasto-strain"] = df["Elasto(90)"]/200000
        return {'times': times, 'dataframe': df}
    
    def split_df_static_pulling(self, intervals=None):
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
        df= self.data_pulling
        df = df[["Force(100)"]].dropna()

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
            df_subset = df.loc[(df.index > start) & (df.index < end),"Force(100)"]
            maximum_idx = np.argmax(df_subset)
            t = {'maximum': df_subset.index[maximum_idx]}

            df_subset = df.loc[(df.index > start) & (df.index < t['maximum']),"Force(100)"]
            idxmin = df_subset.idxmin()
            t['minimum'] = idxmin

            time = time + [t]
        return time

m = DynatreeStaticMeasurment(
    day="2022-04-05", tree="JD18", measurement="M01", pullNo=1)


m.measurement
# %%
m.find_intervals_to_split_measurements_from_csv()

# %%
m.split_df_static_pulling()

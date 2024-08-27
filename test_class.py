#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 12:35:00 2024

@author: marik
"""

import lib_dynatree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import logging

lib_dynatree.logger.setLevel(logging.DEBUG)


def read_tree_configuration():
    file_path = "../data/Popis_Babice_VSE_13082024.xlsx"
    sheet_name = "Prehledova tabulka_zakludaje"
    
    # Načtení dat s vynecháním druhého řádku a nastavením sloupce D jako index
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        skiprows=[1],  # Vynechání druhého řádku
        index_col=0,   # Nastavení čtvrtého sloupce (D) jako index
        nrows=14,       # Načtení 13 řádků s daty
        usecols="D,G,H,I,K,M",  # Načtení pouze sloupců D, G, H, K, L
    )
    
    df.columns=["angle_of_anchorage", "distance_of_anchorage",
             "height_of_anchorage", "height_of_pt",
             "height_of_elastometer"]
    
    return df
DF_PT_NOTES = read_tree_configuration()

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


class DynatreeStaticMeasurment(lib_dynatree.DynatreeMeasurement):


    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    def _find_intervals_to_split_measurements_from_csv(self, csv="csv/intervals_split_M01.csv"):
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

    def _split_df_static_pulling(self, intervals='auto'):
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
        if intervals == 'auto':
            intervals = self._find_intervals_to_split_measurements_from_csv()
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

    @staticmethod
    def _restrict_dataframe(df, column="Force(100)", restricted=(0.3,0.9)):
        """
        Restcts dataframe according to the values in given column.
        
        restricted is a tuple of two numbers.
        It is intended as lower and upper limit for data cut.
        If no restriction is required, use None.
        Default is from 30% to 90% of Fmax
        Used to focus on the interval of interest when processing static pulling data.         
        """
        if restricted == None:
            return df.index[0],df.index[-1]
        minimal_fraction, maximal_fraction = restricted
        maximum = df[column].iat[-1]
        minimum = df[column].iat[0]
        if np.isnan(minimum):
            minimum = 0
        mask = df[column] > maximal_fraction*(maximum-minimum)+minimum
        upper = mask.idxmax()
        subdf = df.loc[:upper, :]
        mask = subdf[column] < minimal_fraction*(maximum-minimum)+minimum
        lower = mask.iloc[::-1].idxmax()
        return lower,upper

    @lib_dynatree.timeit
    def get_static_pulling_data(self, optics=False, restricted=(0.3,0.9)):
        """
        Uniform method to find the data. The data ara obtained from pulling tests
        for M01 and from parquet files for the other measurements.
        
        The data from M01 measurement are from the device output. 
        
        If optics is True, the other measurements are taken from the data
        interpolated to the optics times. 
        
        If optics is True, the data for M02 and higher are from 
        parquet_add_inclino.py library. These data are synchronized with optics and
        recaluculated to the same time index as optics. In this case (Pt3,Y0)
        and (Pt4,Y0) are added as two additional dolumns of the dataframe.
        
        restricted is a tuple of two numbers.
        It is intended as lower and upper limit for data cut.
        If no restriction is required, use None.
        Default is from 30% to 90% of Fmax
        Used to focus on the interval of interest when processing static pulling data.         
        """
        if  (not optics) or (self.measurement == "M01"):
            df = self.data_pulling_interpolated
            if self.measurement == "M01":
                times = self._split_df_static_pulling()
            else:
                times = [{"minimum":0, "maximum": self.data_pulling["Force(100)"].idxmax()}]                
        else:
            dfA = self.data_optics_extra
            dfB = self.data_optics_pt34.loc[:,[("Pt3","Y0"),("Pt4","Y0")]].copy()
            dfB = dfB - dfB.iloc[0,:]
            df = pd.concat([dfA,dfB], axis=1)
            times = [{"minimum":0, "maximum": df["Force(100)"].idxmax().iloc[0]}]
            df = df.drop([i for i in df.columns if i[1]=='X0'], axis=1)
            df.columns = [i[0] for i in df.columns]
        df["Elasto-strain"] = df["Elasto(90)"]/200000
        df_list = [df.loc[t['minimum']:t['maximum'],:] for t in times]
        if restricted is None:
            return df_list
        for i,df in enumerate(df_list):
            lower, upper = DynatreeStaticMeasurment._restrict_dataframe(df, column="Force(100)", restricted=restricted)
            df_list[i] = df.loc[lower:upper,:]
        return df_list
    
    def plot(self):
        fig, ax = plt.subplots()
        df = self.data_pulling_interpolated
        ax.plot(df["Force(100)"])
        for i, j in zip(
                self.get_static_pulling_data(restricted=None),
                self.get_static_pulling_data()
                ):
            ax.plot(i["Force(100)"])
            ax.plot(j["Force(100)"], lw=4)
        ax.legend(["Síla", "Náběh síly", "Rozmezí 30 až 90\nprocent max."])
        ax.set(xlabel="Time", ylabel="Force",
               title=f"Static {self.day} {self.tree} {
                   self.measurement} {self.measurement_type}"
               )
        return fig
    
class DynatreeStaticPulling:
    def __init__(self, data):
        self.data = data
        self.process_inclinometers_major_minor()
    
    def process_inclinometers_major_minor(self):
        """
        The input is dataframe loaded by pd.read_csv from pull_tests directory.
        
        * Converts Inclino(80) and Inclino(81) to blue and yellow respectively.
        * Evaluates the total angle of inclination from X and Y part
        * Adds the columns with Major and Minor axis.
        
        Returns new dataframe with the corresponding columns
        """
        df = pd.DataFrame(index=self.data.index, columns=["blue","yellow"], dtype=float)
        df[["blueX","blueY","yellowX","yellowY"]] = self.data[["Inclino(80)X","Inclino(80)Y", "Inclino(81)X","Inclino(81)Y",]]
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
        self.data = pd.concat([self.data, df], axis=1)
        return self.data

    def process_forces(
            self,
            height_of_anchorage=None,
            rope_angle=None,
            height_of_pt=None, 
            height_of_elastometer=None
            ):
        """
        Input is a dataframe with Force(100) column. Evaluates horizontal and vertical 
        component of the force and moments of force
        """
        df = pd.DataFrame(index=self.data.index)
        # evaluate the horizontal and vertical component
        if rope_angle is None:
            # If rope angle is not given, use the data from the table
            rope_angle = self.data['RopeAngle(100)']
        # evaluate horizontal and vertical force components and moment
        # obrat s values je potreba, protoze data maji MultiIndex
        # shorter names
        df.loc[:,['F_horizontal']] = (self.data['Force(100)'] * np.cos(np.deg2rad(rope_angle))).values
        df.loc[:,['F_vertical']] = (self.data['Force(100)'] * np.sin(np.deg2rad(rope_angle))).values
        df.loc[:,['M']] = df['F_horizontal'] * height_of_anchorage
        df.loc[:,['M_Pt']] = df['F_horizontal'] * ( height_of_anchorage - height_of_pt )
        df.loc[:,['M_Elasto']] = df['F_horizontal'] * ( 
                    height_of_anchorage - height_of_elastometer )
        df.loc[:,["Angle"]] = rope_angle
        return df
    
   
    def get_computed_data(self, optics=False, df_pt_notes=DF_PT_NOTES):
        """
        Gets the data from process_inclinometers_major_minor and from
        process_forces functions.
        
        out is the output of get_static_pulling_data(data_obj)
        """
        day, tree, measurement = self.day, self.tree, self.measurement
        tree = tree[-2:]
        if out is None:
            out = get_static_pulling_data(data_obj,skip_optics=skip_optics)
    
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
    

m = DynatreeStaticMeasurment(
    day="2022-04-05", 
    tree="BK04", 
    measurement="M03"
    )

m.plot()
m.measurement
msplit = m.get_static_pulling_data()
pull = DynatreeStaticPulling(data = msplit[0])
treeNo = int(m.tree[-2:])
a = {}
a['Measure'] = pull.process_forces(
    height_of_anchorage=DF_PT_NOTES.at[treeNo,'height_of_anchorage'],
    height_of_pt=DF_PT_NOTES.at[treeNo,'height_of_pt'],
    rope_angle=DF_PT_NOTES.at[treeNo,'angle_of_anchorage'],
    height_of_elastometer=DF_PT_NOTES.at[treeNo,'height_of_elastometer']
    )
a['Measure'].columns = [f"{i}_Measure" for i in a['Measure'].columns]
a['Rope'] = pull.process_forces(
    height_of_anchorage=DF_PT_NOTES.at[treeNo,'height_of_anchorage'],
    height_of_pt=DF_PT_NOTES.at[treeNo,'height_of_pt'],
    height_of_elastometer=DF_PT_NOTES.at[treeNo,'height_of_elastometer']
    )
a['Rope'].columns = [f"{i}_Rope" for i in a['Rope'].columns]
a = pd.concat(list(a.values()))
ax = a.plot(y=["F_horizontal_Rope", "F_horizontal_Measure"],style = '.')

# %%
ax = a.plot(y=["Angle_Measure","Angle_Rope"], style='.')
ax.grid()
# m.data
# # %%
# a = m.get_static_pulling_data(optics=True)
# b = m.get_static_pulling_data(optics=False)
# ax = b[0].plot(y="Force(100)", style='.')
# a[0].plot(y="Force(100)",ax=ax, style='.')

# # %%
# a[0]["Force(100)"] 
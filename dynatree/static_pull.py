"""
Created on Wed Aug 14 13:25:19 2024

Inputs the pulling data. Raw pullling data for M1, preprocessed pulling data 
for M2 and higher (parquet_add_inclino.py script).

Used in solara_tahovky.py  vit visualizations.

Used to get regressions between variables in the phase of static pulling.

@author: marik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import linregress
from scipy.interpolate import interp1d
from functools import cached_property
from dynatree.find_measurements import get_all_measurements
import dynatree.dynatree as dynatree
import logging
dynatree.logger.setLevel(logging.INFO)

import dynatree.multi_handlers_logger as mhl
import config
from parallelbar import progress_map

def read_tree_configuration():
    file_path = f"csv/angles_measured.csv"

    # Načtení dat s vynecháním druhého řádku a nastavením sloupce D jako index
    df = pd.read_csv(file_path)
    
    df.columns=["state", "day", "tree", "type", "angle_of_anchorage", "diameter",
             "height_of_anchorage", "height_of_pt",
             "height_of_elastometer","kamera","nokamera"]

    df = df.set_index(["tree","day","type"])
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


class DynatreeStaticMeasurement(dynatree.DynatreeMeasurement):
    """
    Like DynatreeMeasurement, adds more properties and methods.
    
    pullings: list of objects DynatreeStaticPulling. This list contains the data
    of the required part of the experiment. Length typically 3 for M01 and 1 for 
    other measurements.
    """
    
    def __init__(self, *args, restricted=(0.3,0.9), optics=False, manual_restrictions=None, **kwargs):
        super().__init__(**kwargs)
        self.optics = optics
        self.restricted = restricted
        self.manual_restrictions = manual_restrictions
        self.parent = dynatree.DynatreeMeasurement(**kwargs)

    @cached_property    
    def pullings(self): 
        return [
            DynatreeStaticPulling(i, tree=self.tree, 
                  measurement_type=self.measurement_type, day=self.day,
                  parent_experiment=self,
                  extra_columns={"blue":"Inclino(80)", "yellow":"Inclino(81)",
                 **self.identify_major_minor}) 
            for i in self._get_static_pulling_data(
                optics=self.optics, restricted=self.restricted)
            ]
    @cached_property
    def regresions(self):
        dynatree.logger.debug("Calculating regressions for static measurement")
        if self.optics and not self.is_optics_available:
            dynatree.logger.info(f"{self.date} {self.tree} {self.measurement}: Optics used but optics is not available.")
            return None
        ans = pd.concat(
            [
                # Do regressions and add columns 
                # pull, optics, lower_bound, upper_bound
                pd.concat(
                    [i.regressions,
                     pd.DataFrame(
                         index=i.regressions.index, 
                         data=np.full_like(i.regressions.index, n), 
                         columns=["pull"], dtype=int)
                     ], axis=1)
                for n, i in enumerate(self.pullings)
            ])
        ans.loc[:,"optics"] = self.optics
        ans.loc[:,["lower_bound","upper_bound"]] = self.restricted
        return ans
    
    @property
    def release_time_force(self):
        """
        For M01 returns the maximum of the Force during the first pull.
        
        For other cases returns the same property of the parent class.
        """
        if self.measurement != "M01":
            return self.parent.release_time_force
        times = self._split_df_static_pulling()
        data = self.data_pulling["Force(100)"]
        return data.loc[:times[1]['minimum']].idxmax()

    @property
    def release_time_optics(self):
        if self.measurement != "M01":
            return self.parent.release_time_optics
        times = self._split_df_static_pulling()
        data = self.data_optics[("Pt3","Y0")]
        data = data - data.iloc[0]
        data = (data
                .loc[:times[1]['minimum']]
                .pipe(np.abs)
                )
        return data.idxmax()

    def __str__(self):
        ans = super(DynatreeStaticMeasurement,self).__str__()
        ans = ans + f"\nStatic options: optics={self.optics}, restricted={self.restricted}, manual restrictions={self.manual_restrictions}"
        ans = ans + f"\nOptics availability: {self.is_optics_available}"
        return ans
    

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

    def _split_df_static_pulling(self, intervals='auto', probe="Force(100)"):
        """
        Analyzes data in static tests, with three pulls. Inputs the dataframe, 
        outputs the dictionary. 
        
        output['times'] contains the time intervals of increasing pulling force. 
        
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
        if probe in ["Pt3","Pt4"]:
            df = self.data_optics
            df = df[[(probe,"Y0")]].dropna()
            df.columns = [probe]
            df = df - df.iloc[0,:]
            df = df * np.sign(df.sum())
        else:
            df= self.data_pulling
            df = df[[probe]].dropna()

        # Interpolace a vyhlazeni
        new_index = np.arange(df.index[0],df.index[-1],0.1)
        window_length = 100
        polyorder = 3
        newdf = df[[probe]].dropna()
        interpolation_function = interp1d(newdf.index, newdf[probe], kind='linear')
        df_interpolated = pd.DataFrame(interpolation_function(new_index), index=new_index, columns=[probe])
        df_interpolated['probe_smoothed'] = savgol_filter(df_interpolated[probe], window_length=window_length, polyorder=polyorder)

        steps_down=None
        if intervals is None:
            # Divide domain into interval where force is large and small
            maximum = df_interpolated[probe].max()
            df_interpolated["probe_step"] = (df_interpolated[probe]>0.5*maximum).astype(int)

            # identify jumps down
            diff_d1 = df_interpolated["probe_step"].diff()
            steps_down = list(diff_d1.index[diff_d1<0])
            intervals = zip([0]+steps_down[:-1],steps_down)

        time = []
        for start,end in intervals:
            df_subset = df.loc[(df.index > start) & (df.index < end),probe]
            maximum_idx = np.argmax(df_subset)
            t = {'maximum': df_subset.index[maximum_idx]}

            df_subset = df.loc[(df.index > start) & (df.index < t['maximum']),probe]
            idxmin = df_subset.idxmin()
            t['minimum'] = idxmin

            time = time + [t]
        return time

    @staticmethod
    def _restrict_dataframe(df, column="Force(100)", restricted=(0.3,0.9), manual_restrictions=None):
        """
        Restricts dataframe according to the values in given column.
        
        restricted is a tuple of two numbers.
        It is intended as lower and upper limit for data cut.
        If no restriction is required, use None.
        Default is from 30% to 90% of Fmax
        Used to focus on the interval of interest when processing static pulling data.

        Returns lower and upper bound in time scale.
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

    # @dynatree.timeit
    def _get_static_pulling_data(self, optics=False, restricted=(0.3,0.9)):
        """
        Uniform method to extract the data. The data are obtained 
        from parquet files for the other measurements.
        
        The data from M01 measurement are from the device output. 
        
        If optics is True, the other measurements are taken from the 
        data interpolated to the optics times. 
        
        If optics is True, the data for M02 and higher are from 
        parquet_add_inclino.py library. These data are synchronized 
        with optics and recaluculated to the same time index 
        as optics. In this case (Pt3,Y0)
        and (Pt4,Y0) are added as two additional dolumns of the
        dataframe.
        
        restricted is a tuple of two numbers.
        It is intended as lower and upper limit for data cut.
        If no restriction is required, use None.
        Default is from 30% to 90% of Fmax
        Used to focus on the interval of interest when processing static pulling data.         
        
        If restricted is get_all, return the whole dataset.
        """
        if optics and not self.is_optics_available:
            # dynatree.logger.warning(f"Optics not available for {self.day} {self.tree} {self.measurement}")
            return []
        if not optics:
            df = self.data_pulling_interpolated
        else:
            dfA = self.data_optics_extra
            dfB = self.data_optics_pt34.loc[:,[("Pt3","Y0"),("Pt4","Y0")]].copy()
            dfB = dfB - dfB.iloc[0,:]
            df = pd.concat([dfA,dfB], axis=1)
            df = df.drop([i for i in df.columns if i[1]=='X0'], axis=1)
            df.columns = [i[0] for i in df.columns]

        if  self.measurement == "M01":
            times = self._split_df_static_pulling()
        else:
            if self.data_pulling["Force(100)"].isna().all():
                maximum = 0
            else:
                maximum = self.data_pulling["Force(100)"].idxmax()
            times = [{"minimum":0, "maximum": maximum}]


        df["Elasto-strain"] = df["Elasto(90)"]/200000
        if restricted == 'get_all':
            return df
        df_list = [df.loc[t['minimum']:t['maximum'],:] for t in times]
        if restricted is None:
            return df_list
        for i,df in enumerate(df_list):
            lower, upper = DynatreeStaticMeasurement._restrict_dataframe(df, column="Force(100)", restricted=restricted)
            df_list[i] = df.loc[lower:upper,:]
        # df_list = [DynatreeStaticMeasurement.add_colors(df) for df in df_list]
        return df_list
    
    # @staticmethod
    # def add_colors(df):
    #     """
    #     Adds new columns to dataframe by replacing Inclino(80) and 
    #     Inclino(81) by colors blue and yellow.
    #     """
    #     cols = df.columns
    #     for col in cols:
    #         if "Inclino" in col:
    #             newcol = col
    #             newcol = newcol.replace("Inclino(80)","blue")
    #             newcol = newcol.replace("Inclino(81)","yellow")
    #             df[newcol] = df[col]
    #     return df
    
    def plot(self):
        fig, ax = plt.subplots()
        df = self.data_pulling_interpolated
        if df is None:
            return None
        ax.plot(df["Force(100)"])
        for i, j in zip(
                self._get_static_pulling_data(restricted=None, optics=self.optics),
                self._get_static_pulling_data(optics=self.optics)
                ):
            ax.plot(i["Force(100)"])
            ax.plot(j["Force(100)"], lw=4)
        ax.legend(["Síla", "Náběh síly", "Rozmezí 30 až 90\nprocent max."])
        ax.set(xlabel="Time", ylabel="Force",
               title=f"Static {self.day} {self.tree} {self.measurement} {self.measurement_type}"
               )
        return fig

class DynatreeStaticPulling:
    """
    One pulling phase of the experiment. The data are already restricted
    the the phase we are interested in. The variables under consideration 
    are evaluated during initilaization. Also regressions are evaluated 
    automatically.
    
    The plot method plots the data and the regression, if available.
    
    If ini_forces or ini_regress are False, do not evaluate the forces
    and regressions. In this case the variable tree is not important.
    """
    def __init__(self, data_, tree=None, ini_forces=True, ini_regress=True, measurement_type="normal", day=None, extra_columns=None, parent_experiment=None):
        data = data_.copy()
        if extra_columns is not None:
            for i in extra_columns.keys():
                data.loc[:,i] = data.loc[:,extra_columns[i]]
        self.data = data
        # self._process_inclinometers_major_minor()
        self.day = day
        self.tree = tree
        self.measurement_type = measurement_type
        self.parent_experiment = parent_experiment
        if ini_forces:
            pars = dict(height_of_anchorage = DF_PT_NOTES.at[(tree,day,measurement_type),'height_of_anchorage'],
                height_of_pt = DF_PT_NOTES.at[(tree,day,measurement_type),'height_of_pt'],
                rope_angle = DF_PT_NOTES.at[(tree,day,measurement_type),'angle_of_anchorage'],
                height_of_elastometer = DF_PT_NOTES.at[(tree,day,measurement_type),'height_of_elastometer'],
                suffix = "")
            self._process_forces(**pars)
            # self._process_forces(
            #     height_of_anchorage= DF_PT_NOTES.at[treeNo,'height_of_anchorage'],
            #     height_of_pt= DF_PT_NOTES.at[treeNo,'height_of_pt'],
            #     rope_angle = rope_angle(measurement_type=self.measurement_type, tree=self.tree, day=self.day, source='pulling_tests'),
            #     height_of_elastometer= DF_PT_NOTES.at[treeNo,'height_of_elastometer'],
            #     suffix = "Rope"
            #     )
        if ini_regress:
            self.regressions = self._get_regressions_for_one_pull()

    def __str__(self):
        return f"Dynatree static pulling, data shape {self.data.shape}"
    
    def __repr__(self):
        return f"Dynatree static pulling, data shape {self.data.shape}"
    
    # def _process_inclinometers_major_minor(self):
    #     """
    #     The input is dataframe loaded by pd.read_csv from pull_tests directory.
        
    #     * Converts Inclino(80) and Inclino(81) to blue and yellow respectively.
    #     * Ads blueMaj etc. as a copy of blueX or blueY, respectively.
        
    #     Returns new dataframe with the corresponding columns
    #     """
    #     df = pd.DataFrame(index=self.data.index, columns=["blue","yellow"], dtype=float)
    #     df[["blueX","blueY","yellowX","yellowY"]] = self.data[["Inclino(80)X","Inclino(80)Y", "Inclino(81)X","Inclino(81)Y",]]
    #     for inclino in ["blue","yellow"]:
    #         df.loc[:,[inclino]] = arctand(
    #             np.sqrt((tand(df[f"{inclino}X"]))**2 + (tand(df[f"{inclino}Y"]))**2 )
    #             )
    #         # najde maximum bez ohledu na znamenko
    #         maxima = df[[f"{inclino}X",f"{inclino}Y"]].abs().max()
    #         # vytvori sloupce blue_Maj, blue_Min, yellow_Maj,  yellow_Min....hlavni a vedlejsi osa
    #         if maxima[f"{inclino}X"]>maxima[f"{inclino}Y"]:
    #             df.loc[:,[f"{inclino}_Maj"]] = df[f"{inclino}X"]
    #             df.loc[:,[f"{inclino}_Min"]] = df[f"{inclino}Y"]
    #         else:
    #             df.loc[:,[f"{inclino}_Maj"]] = df[f"{inclino}Y"]
    #             df.loc[:,[f"{inclino}_Min"]] = df[f"{inclino}X"]
    #         # Najde pozici, kde je extremalni hodnota - nejkladnejsi nebo nejzapornejsi
    #         idx = df[f"{inclino}_Maj"].abs().idxmax()
    #         # promenna bude jednicka pokus je extremalni hodnota kladna a minus
    #         # jednicka, pokud je extremalni hodnota zaporna
    #         if pd.isna(idx):
    #             znamenko = 1
    #         else:
    #             znamenko = np.sign(df[f"{inclino}_Maj"][idx])
    #         # V zavisosti na znamenku se neudela nic nebo zmeni znamenko ve sloupcich
    #         # blueM, blueV, yellowM, yellowV
    #         for axis in ["_Maj", "_Min"]:
    #             df.loc[:,[f"{inclino}{axis}"]] = znamenko * df[f"{inclino}{axis}"]
    #     self.data = pd.concat([self.data, df], axis=1)
    #     return self.data

    def _process_forces(
            self,
            height_of_anchorage=None,
            rope_angle=None,
            height_of_pt=None, 
            height_of_elastometer=None,
            suffix = ""
            ):
        """
        Input is a dataframe with Force(100) column. Evaluates horizontal and vertical 
        component of the force and moments of force
        """
        df = pd.DataFrame(index=self.data.index)
        # evaluate horizontal and vertical force components and moment
        # obrat s values je potreba, protoze data maji MultiIndex
        # shorter names
        if suffix != "":
            suffix = f"_{suffix}"
        df.loc[:,['F_horizontal']] = (self.data['Force(100)'] * np.cos(np.deg2rad(rope_angle))).values
        df.loc[:,['F_vertical']] = (self.data['Force(100)'] * np.sin(np.deg2rad(rope_angle))).values
        df.loc[:,['M']] = df['F_horizontal'] * height_of_anchorage
        df.loc[:,['M_Pt']] = df['F_horizontal'] * ( height_of_anchorage - height_of_pt )
        df.loc[:,['M_Elasto']] = df['F_horizontal'] * ( 
                    height_of_anchorage - height_of_elastometer )
        df.loc[:,["Angle"]] = rope_angle
        df.columns = [f"{i}{suffix}" for i in df.columns]
        # self.data = pd.concat([self.data, df], axis=1)
        self.data.loc[:,df.columns] = df
        return self.data
    
    def plot(self, pullNo=None):
        f,a = plt.subplots(2,2,
                           figsize=(12,9)
                           )
        a = a.reshape(-1)
        df = self.data

        df.loc[:,["Force(100)"]].plot(ax=a[0], legend=False, xlabel="Time", ylabel="Force", style='.')
        colors = ["blue","yellow"]

        df.loc[:,["blueMaj", "blueMin", "yellowMaj", "yellowMin"]
               ].plot(ax=a[2], xlabel="Time", style='.')
        a[2].grid()
        a[2].set(ylabel="Angle")
        a[2].legend(title="Inclinometers axes")

        df.plot(x="M_Elasto", y="Elasto-strain", ax=a[1], xlabel="Momentum at extensometer", ylabel="Strain on extensometer", style='.', color="C1", legend=None)
        a[1].grid()

        a[3].plot(df["M"], df[colors], '.')
        a[3].set(
                    xlabel = "Momentum from Measured rope angle",
                    ylabel = "Angle",
                    )
        a[3].legend(colors, title="Inclinometers")

        for _ in [a[0],a[3]]:
            _.set(ylim=(0,None))
            _.grid()
        if pullNo != None:
            f.suptitle(f"Detail, pull {pullNo}")
        plt.tight_layout()
        return f

    
    def _get_regressions_for_one_pull(self):
        """
        Get regressions for one measurement. 
        """
        if "Pt3" in self.data.columns:
            pt_reg = [
                ["M_Pt","Pt3", "Pt4"],
                ]
        else:
            pt_reg = []
        reg = DynatreeStaticPulling._get_regressions(self.data,
            [
            ["M", "blueMaj", "yellowMaj"],
            ["M_Elasto", "Elasto-strain"],
            ["Force(100)", "blueMaj", "yellowMaj", "Elasto(90)"],
            ]+pt_reg, msg=f"{self.parent_experiment.parent if self.parent_experiment  is not None else None}"
            )
        return reg
    
    @staticmethod
    def _get_regressions(df, collist, msg=""):
        """
        Return regression in dataframe. 
        
        The variable colist is list. This variable is assumed to be list of lists. 
        In each sublist, the regression of the first item to the other ones is 
        evaluated.
        """
        data = [DynatreeStaticPulling._get_regressions_for_one_column(
            df.loc[:, i], i[0], msg=msg) for i in collist]
        return pd.concat(data)
    
    @staticmethod
    def _get_regressions_for_one_column(df, independent, msg=""):
        regrese = {}
        dependent = [_ for _ in df.columns if _ !=independent]
        dynatree.logger.debug(
            f"Regressions on dataframe of shape {df.shape}\n    independent {independent}, dependent {dependent}")
        for i in dependent:
            # remove nan valules, if any
            cleandf = df.loc[:,[independent,i]].dropna()
            # do regresions without nan

            try:
                reg = linregress(cleandf[independent],cleandf[i])
                regrese[i] = [independent, i, reg.slope, reg.intercept, reg.rvalue ** 2, reg.pvalue, reg.stderr,
                              reg.intercept_stderr]
            except:
                dynatree.logger.error(f"Linear regression failed for {independent} versus {i}. {msg}")
                pass

        ans_df = pd.DataFrame(regrese,
                              index=["Independent", "Dependent", "Slope", "Intercept", "R^2", "p-value", "stderr",
                                     "intercept_stderr"], columns=dependent).T
        return ans_df    

def proces_one_row(row):
    ans = {}
    day = row['day']
    tree = row['tree']
    measurement = row['measurement']
    optics = row['optics']
    measurement_type = row['type']
    for cut in [.30]:
        for use_optics in [False, True]:
            # try:
                # get regressions for two cut-out values and merge
            data_obj = DynatreeStaticMeasurement(day=day, tree=tree, measurement=measurement,
                                                 measurement_type=measurement_type, optics=use_optics,
                                                 restricted=(cut,0.9))
            if data_obj.parent.data_pulling is None:
                logger.warning(f"There are no data for pulling tests for this case: {day} {tree} {measurement}, {measurement_type}.")
                continue
            for i,pull in enumerate(data_obj.pullings):
                regressions = pull.regressions
                regressions["optics"] = use_optics
                regressions["lower_cut"] = cut
                regressions["upper_cut"] = 0.9
                regressions["day"] = day
                regressions["tree"] = tree
                regressions["measurement"] = measurement
                regressions["type"] = measurement_type
                regressions["pullNo"] = i
                ans[(use_optics,cut,i, day, tree, measurement, measurement_type)] = regressions
    try:
        ans = pd.concat(ans)
    except:
        ans = pd.DataFrame()
        dynatree.logger.error(f"Failed to process {day} {tree} {measurement} {measurement_type}.")
    return ans

logger = mhl.setup_logger(prefix="static_pull_")
logger.setLevel(logging.INFO)
logger.info("========== INITIALIZATION OF static-pull.py  ============")

def main():
    df = get_all_measurements(method='all', type='all')
    # drop missing optics
    # df = df[~((df["day"]=="2022-04-05")&(df["tree"]=="BK21")&(df["measurement"]=="M5"))]
    # df = df.iloc[:3,:]
    # for _,row in df.iterrows():
    #     proces_one_row(row)
    # return
    ans = progress_map(proces_one_row, [i for _,i in df.iterrows()])
    ans = pd.concat(ans)
    df_all_data = ans.reset_index(drop=True)
    return df_all_data

if __name__ == "__main__":
    # day, tree, measurement, mt = "2022-08-16", "BK08", "M02", "normal"
    # day,tree,measurement, mt = "2023-07-17", "BK01", "M01", "afterro"
    # rope_angle(mt, day, tree, 'pulling_tests')

    # day, tree, measurement, mt = "2021-06-29", "BK08", "M04", "normal"
    # m = DynatreeStaticMeasurement(
    #     day=day, tree=tree, measurement=measurement, measurement_type=mt, optics=True)
    # import rich
    # for pull in m.pullings:
    #     rich.print(pull.data.columns)
    #     rich.print(pull.regressions)
    # m.plot()
    # for n,i in enumerate(m.pullings):
    #     i.plot(n)
        
        
    # ans = process_data(data_obj, skip_optics=True)

    # ans_10 = get_regressions_for_one_measurement(data_obj,minimal_fraction=0.1, skip_optics=False)
    # ans_30 = get_regressions_for_one_measurement(day, tree, measurement,minimal_fraction=0.3)
    # ans = pd.concat([ans_10,ans_30])
    # main_nakresli()
    
    # These two lines are for production code to do final analysis
    main_output = main()
    main_output.to_csv("csv_output/regressions_static.csv")

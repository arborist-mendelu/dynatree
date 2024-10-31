import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import linregress
import os

from solara.website.pages.documentation.examples.general.deploy_model import slope

pd.options.plotting.backend = "plotly"

DIRECTORY = '../data/ema'

def read_csvdata_inclinometers(file):
    """
    Read data from pulling tests. Used to save parquet files.
    """
    df_pulling_tests = pd.read_csv(
        file,
        skiprows=55,
        decimal=",",
        sep=r'\s+',
        skipinitialspace=True,
        na_values="-"
        )
    df_pulling_tests["Time"] = df_pulling_tests["Time"] - df_pulling_tests["Time"][0]
    df_pulling_tests.set_index("Time", inplace=True)
    df_pulling_tests = df_pulling_tests.drop(['Nr', 'Year', 'Month', 'Day'], axis=1)
    return df_pulling_tests

class PullingTest:
    def __init__(self, file, directory='data', localfile=True):
        if localfile:
            self.filename = directory+"/"+file
            self.data = read_csvdata_inclinometers(self.filename)
        else:
            self.data = read_csvdata_inclinometers(file)
        self.measurement = file.replace(".TXT","")
    @property
    def major_inclinometers(self):
        df_majorminor = major_minor_axes()
        inclinometers = df_majorminor.loc[self.measurement, :]
        return inclinometers

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
        df= self.data
        df = df[[probe]].dropna()

        # Interpolace a vyhlazeni
        new_index = np.arange(df.index[0],df.index[-1],0.1)
        window_length = 200
        polyorder = 1
        newdf = df[[probe]].dropna()
        interpolation_function = interp1d(newdf.index, newdf[probe], kind='linear')
        df_interpolated = pd.DataFrame(interpolation_function(new_index), index=new_index, columns=[probe])
        df_interpolated['probe_smoothed'] = savgol_filter(df_interpolated[probe], window_length=window_length, polyorder=polyorder)

        steps_down=None
        if True or intervals is None:
            # Divide domain into interval where force is large and small
            maximum = df_interpolated[probe].max()
            df_interpolated["probe_step"] = (df_interpolated["probe_smoothed"]>0.4*maximum).astype(int)

            # identify jumps down
            diff_d1 = df_interpolated["probe_step"].diff()
            steps_down = list(diff_d1.index[diff_d1<0])
            intervals = zip([0]+steps_down[:-1],steps_down)
        time = []
        for start,end in intervals:
            if end-start < 3:
                continue
            df_subset = df.loc[(df.index > start) & (df.index < end),probe]
            maximum_idx = np.argmax(df_subset)
            t = {'maximum': df_subset.index[maximum_idx]}

            df_subset = df.loc[(df.index > start) & (df.index < t['maximum']),probe]
            idxmin = df_subset.idxmin()
            t['minimum'] = idxmin

            time = time + [t]
        return time

    def intervals_of_interest(self, probe="Force(100)"):
        intervals = self._split_df_static_pulling()
        df = self.data
        ans = []
        for i in intervals:
            minimum = df.loc[i['minimum'],probe]
            maximum = df.loc[i['maximum'],probe]
            upper_bound = 0.9*(maximum-minimum)+minimum
            lower_bound = 0.3*(maximum-minimum)+minimum
            lb = np.argmax(df.loc[i['minimum']:i['maximum'],probe]>lower_bound)
            ub = np.argmax(df.loc[i['minimum']:i['maximum'],probe]>upper_bound)
            ans = ans + [[df.loc[i['minimum']:i['maximum'],probe].index[lb], df.loc[i['minimum']:i['maximum'],probe].index[ub]]]
        return ans


def major_minor_axes():
    df_majorminor = pd.read_csv(f'{DIRECTORY}/ema-tahovky-major.csv', index_col=0, sep=";")

    # Funkce pro úpravu buňky do formátu "Inclino(column_letter)"
    def update_cell(value, column):
        return f"Inclino({column}){value}" if pd.notna(value) else value

    # Použití funkce na každý sloupec a buňku
    for column in df_majorminor.columns:
        df_majorminor[column] = df_majorminor[column].apply(lambda x: update_cell(x, column))
    return df_majorminor

def slopes():
    files = [f.replace(".TXT", "") for f in os.listdir(DIRECTORY) if
             os.path.isfile(os.path.join(DIRECTORY, f)) and 'TXT' in f]
    files.sort()
    df_majorminor = major_minor_axes()

    ans = {}
    for file in files:
        t = PullingTest(file+".TXT", directory=DIRECTORY, localfile=True)
        df = t.data.loc[:,["Force(100)",*t.major_inclinometers]]
        df = df.interpolate(method='index').dropna().abs()
        intervals = t.intervals_of_interest()
        for i,interval in enumerate(intervals):
            # print(interval)
            subdf = df.loc[interval[0]:interval[1],:]
            # print(subdf.head())
            for column in subdf.columns:
                if column=="Force(100)":
                    continue
                slope, intercept, r, *a = linregress(subdf.loc[:,"Force(100)"], subdf.loc[:,column])
                ans[(file, i, column)] = slope, r**2
    ans = pd.DataFrame.from_dict(ans, orient='index', columns=['slope', 'R^2'])
    return(ans)

def main()
    ans = slopes()
    print(ans)

if __name__ ==  "__main__":
    main()
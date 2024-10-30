import pandas as pd
pd.options.plotting.backend = "plotly"

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

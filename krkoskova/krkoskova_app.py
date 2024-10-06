import matplotlib.pyplot as plt
import numpy as np
import solara
import pandas as pd
import glob
import plotly.express as px
from plotly_resampler import FigureResampler

from scipy.signal import find_peaks
from scipy import signal
from scipy.fft import fft, fftfreq

data_directory = "../data/krkoskova"

names = glob.glob(f"{data_directory}/*acc.parquet")
names = [i.replace("_acc.parquet","").replace(data_directory,"").replace("/","") for i in names]
names.sort()
names = {i.split("_")[0] : [j.split("_")[1] for j in names if i.split("_")[0] in j] for i  in names}



styles_css = """
        .widget-image{width:100%;} 
        .v-btn-toggle{display:inline;}  
        .v-btn {display:inline; text-transform: none;} 
        .vuetify-styles .v-btn-toggle {display:inline;} 
        .v-btn__content { text-transform: none;}
        """
kwds = {"template": "plotly_white"}

trees = list(names.keys())
tree = solara.reactive(trees[0])
measurement = solara.reactive("lano01")
sensor = solara.reactive("ext01")
sensors = ["ext01", "ext02"] + ['A01_z', 'A01_y', 'A02_z', 'A02_y', 'A03_z', 'A03_y']
n = solara.reactive(8)

@solara.component
def plot(df, msg=None, resample=False, title=None, **kw):
    # df = df_.copy()
    fig = px.line(df, title=title, **kwds, **kw)    
    if resample:
        fig_res = FigureResampler(fig)
        solara.FigurePlotly(fig_res)    
    else:
        solara.FigurePlotly(fig)    
    if resample:
        solara.Info(
"""
Data jsou s příliš vysokou samplovací frekvencí. Kvůli plynulosti 
zobrazování jsou zde přesamplována pomocí knihovny `plotly_resampler`. 
Do výpočtů již vstupují data všechna.
""")
    if msg is not None:
        msg

class Measurement:
    def __init__(self, tree, measurement):
        self.tree = tree
        self.measurement = measurement
        
    @property
    def data_extenso(self):
        """
        Returns dataframe with data from extensometers. Sampling is 500Hz.
        """
        try:
            df = pd.read_parquet(f"{data_directory}/{self.tree}_{self.measurement}_ext.parquet")
        except:
            return pd.DataFrame()
        df.index = np.arange(0,len(df))*0.002
        df.index.name = "Time"
        return df

    @property
    def data_acc(self):
        """
        Returns dataframe with data from accelerometers. Sampling is 5000Hz.
        """
        try:
            df = pd.read_parquet(f"{data_directory}/{self.tree}_{self.measurement}_acc.parquet")
        except:
            return pd.DataFrame()
        df.index = np.arange(0,len(df))*0.0002
        df.index.name = "Time"
        return df
    
    @property
    def release_time(self, probe="ext02"):
        if "tuk" in self.measurement:
            return 0
        df = self.data_extenso
        if len(df) == 0:
            return 0
        df = df[probe]
        df = df - df.mean()
        df = df.abs()
        release_time = df.idxmax()
        return release_time
    
    def sensor_sampling(self, sensor):
        if "ext" in sensor:
            return 500
        else:
            return 5000

    def dt(self, sensor):
        if "ext" in sensor:
            return 0.002
        else:
            return 0.0002


    def sensor_data(self, sensor):
        if "ext" in sensor:
            df = self.data_extenso
        else:
            df = self.data_acc
        if sensor in df.columns:
            return df[sensor]
        else:
            return pd.Series()

class Signal():
    def __init__(self, data, time, dt):
        self.data = data
        self.time = time
        self.dt = dt  
        if dt == 0.002:
            self.fs = 500
        elif dt == 0.0002:
            self.fs = 5000
        
    @property
    def fft(self):
        N = len(self.data)  # get the number of points
        xf_r = fftfreq(N, self.dt)[:N//2]
        yf = fft(self.data)  # preform FFT analysis
        yf_r = 2.0/N * np.abs(yf[0:N//2])
        return pd.DataFrame(data=yf_r, index=xf_r)

    def welch(self,n=8):
        nperseg = 2**n
        f, Pxx = signal.welch(x=self.data, fs=self.fs, nperseg=nperseg)
        df_welch = pd.DataFrame(index=f, data=Pxx)
        return df_welch

@solara.component
def Page():
    solara.Style(styles_css)
    solara.Title("Krkoškova")
    with solara.Sidebar():
        solara.Markdown("**Tree**")
        solara.ToggleButtonsSingle(value=tree, values=trees)
        solara.Markdown("**Measurement**")
        solara.ToggleButtonsSingle(value=measurement, values = names[tree.value])
        solara.Markdown("**Sensor**")
        solara.ToggleButtonsSingle(value=sensor, values = sensors)

    m = Measurement(tree.value, measurement.value)
    data = m.sensor_data(sensor.value)
    if len(data) == 0:
        return None

    with solara.lab.Tabs():
        with solara.lab.Tab("Time domain"):
            if data is not None:
                plot(data, resample="A" in sensor.value, title = f"Dataset: {m.tree}, {m.measurement}, {sensor.value}")
                solara.Text(
f"""
Release time is established as a maximum of ext02 for "lano0X" and as zero for "tuk0X".
Here we consider release time {m.release_time}.
""")
        sig = data.loc[m.release_time:]
        sig = sig - sig.mean()
        new_time = np.arange(sig.index[0], sig.index[0]+60,m.dt(sensor.value))
        sig = np.interp(new_time, sig.index, sig, right=0)
        tukey_window = signal.windows.tukey(len(sig), alpha=0.1, sym=False)
        sig_tukey = sig * tukey_window
        s = Signal(sig, new_time, m.dt(sensor.value))

        with solara.lab.Tab("Frequency domain (FFT)"):
            with solara.Card():
                fft_image = s.fft
                plot(fft_image, 
                     resample=False, title = f"FFT for dataset: {m.tree}, {m.measurement}, {sensor.value}",
                     log_y=True, range_x=[0,10]
                     )
            with solara.Card():
                plot(pd.DataFrame(data=sig_tukey, index=new_time), resample=True, title = f"Dataset: {m.tree}, {m.measurement}, {sensor.value}")

        with solara.lab.Tab("Frequency domain (Welch)"):
            with solara.Row():
                solara.Markdown(r"$n$ (where $\text{nperseg}=2^n$)")
                solara.ToggleButtonsSingle(values=list(range(6,13)), value=n)
            welch_image = s.welch(n=n.value)
            plot(welch_image, 
                 resample=False, title = f"Welch spectrum for dataset: {m.tree}, {m.measurement}, {sensor.value}",
                 log_y=True
                 )
            
            
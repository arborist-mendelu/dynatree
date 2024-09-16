import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import plotly.express as px
from scipy.signal import find_peaks
from scipy import signal


# get FFT output
def do_fft(signal_fft, dt):
    N = signal_fft.shape[0]  # get the number of points
    xf_r = fftfreq(N, dt)[:N//2]
    df_fft = pd.DataFrame(index=xf_r, columns=signal_fft.columns)
    signal_fft = signal_fft - np.nanmean(signal_fft) # mean value to zero

    for col in signal_fft.columns:
        yf = fft(signal_fft.loc[:,col].values)  # preform FFT analysis
        yf_r = 2.0/N * np.abs(yf[0:N//2])
        df_fft.loc[:,col] = yf_r
    return df_fft



def do_fft_image(signal_fft, dt, title="", restrict = None):
    df_fft = do_fft(signal_fft, dt)
    if restrict is not None:
        df_fft = df_fft.loc[:restrict,:]
    ymax = df_fft.to_numpy().max()
    figFFT = px.line(df_fft, height = 400, width=1200,
                          title=f"FFT spectrum "+title, 
                          log_y=True, range_x=[0,50], range_y=[ymax/100000, ymax*2]
    )
    figFFT.update_layout(xaxis_title="Freq/Hz", yaxis_title="FFT amplitude")
    return {'fig': figFFT, 'df':df_fft}

def do_welch(signal_welch, nperseg=2**14, fs = 5000):
    f, Pxx = signal.welch(x=signal_welch.iloc[:,0], fs=fs, nperseg=nperseg)
    df_welch = pd.DataFrame(index=f, data=Pxx, columns=[signal_welch.columns[0]])
    if len(signal_welch.columns)==1:
        return df_welch
    for col in signal_welch.columns[1:]:
        f, Pxx = signal.welch(x=signal_welch.loc[:,col], fs=fs, nperseg=nperseg)
        df_welch.loc[:,col] = Pxx
    return df_welch

def do_welch_image(signal_fft, title="", restrict = None, nperseg=2**14, fs = 5000):
    # print("welch enter", signal_fft.columns)
    df_fft = do_welch(signal_fft, nperseg=nperseg, fs=fs)
    # print(df_fft.columns)
    if restrict is not None:
        df_fft = df_fft.loc[:restrict,:]
    ymax = df_fft.to_numpy().max()
    figFFT = px.line(df_fft, height = 400, width=1200,
                          title=f"Power spectrum "+title, 
                          log_y=True, range_x=[0,50], range_y=[ymax/100000, ymax*2]
    )
    figFFT.update_layout(xaxis_title="Freq/Hz", yaxis_title="Power")
    return {'fig': figFFT, 'df':df_fft}

def process_signal(df_, duration=60, start=0, dt=0.0002, tukey=0.1):
    df = df_.copy()
    ans = [
        process_signal_single(df.loc[:,col], duration=duration, start=start, dt=dt, tukey=tukey)
        for col in df.columns
        ]    
    ans = pd.concat(ans, axis=1)
    ans.columns = df.columns
    return ans

def process_signal_single(df_, duration=60, start=0, dt=0.0002, tukey=0.1):
    df = df_.copy()
   
    # 3. Ořízni signál - vše před nejvyšším bodem se odstraní
    df_trimmed = df.loc[start:]
    
    # 4. Ořízni signál na 60 sekund
    time_end = df_trimmed.index[0] + 60
    df_trimmed = df_trimmed.loc[:time_end]
    
    # 5. Vycentruj signál na vodorovnou osu (odečti průměr)
    df_trimmed = df_trimmed - df_trimmed.mean()
    
    # 6. Pokud je signál kratší než 60 sekund, doplň nulami
    required_length = 60
    actual_length = df_trimmed.index[-1] - df_trimmed.index[0]
    delta = required_length - actual_length
    if delta>0:
        # Vytvoření nového časového indexu, který má 60 sekund
        add_index = np.arange(0,delta,dt) + df_trimmed.index[-1] + dt 
        tail = pd.Series(data = 0, index=add_index)
        # tail.loc[:] = 0
        # Doplnění nul, pokud je signál kratší
        
        df_trimmed = pd.concat([df_trimmed, tail])

    tukey_window = signal.windows.tukey(len(df_trimmed), alpha=tukey, sym=False)
    df_trimmed = df_trimmed * tukey_window

    
    return df_trimmed
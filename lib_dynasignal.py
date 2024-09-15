import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import plotly.express as px
from scipy.signal import find_peaks
from scipy import signal


# get FFT output
def do_fft(signal, dt, title=""):
    time_fft = signal.index.values    
    N = time_fft.shape[0]  # get the number of points
    xf_r = fftfreq(N, dt)[:N//2]
    df_fft = pd.DataFrame(index=xf_r, columns=["Amplitude"])
    signal_fft = signal.values
    time_fft = time_fft - time_fft[0]
    signal_fft = signal_fft - np.nanmean(signal_fft) # mean value to zero

    yf = fft(signal_fft)  # preform FFT analysis
    yf_r = 2.0/N * np.abs(yf[0:N//2])
    df_fft["Amplitude"] = yf_r
    ymax = np.max(yf_r)
    figFFT = px.line(df_fft, height = 400, width=1200,
                          title=f"FFT spectrum "+title, 
                          log_y=True, range_x=[0,50], range_y=[ymax/100000, ymax*2]
    )
    figFFT.update_layout(xaxis_title="Freq/Hz", yaxis_title="FFT amplitude")
    return figFFT

def process_signal(df_, duration=60, start=0, dt=0.0002, tukey=0.1):
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
        add_index = np.arange(0,delta,dt) + df_trimmed.index[-1]+dt 
        tail = pd.Series(index=add_index)
        tail.loc[:] = 0
        # Doplnění nul, pokud je signál kratší
        
        df_trimmed = pd.concat([df_trimmed, tail])

    tukey_window = signal.windows.tukey(len(df_trimmed), alpha=tukey, sym=False)
    df_trimmed = df_trimmed * tukey_window

    
    return df_trimmed
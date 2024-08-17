#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:37:18 2024

@author: marik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import FFT_spectrum as fftdt
import lib_dynatree as dt
import lib_streamlit
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objs as go

plt.rcParams["figure.figsize"] = (10,6)
st.set_page_config(layout="wide")

fs = 100 # resampled signal
# files = glob.glob("../01_Mereni_Babice_*_optika_zpracovani/csv/*")
# data = [{'year':i[24:28], 'month': i[22:24], 'day': i[20:22], 'measurement': i[-6:-4], 'tree':i[-10:-8],
#          'date': f"{i[24:28]}-{i[22:24]}-{i[20:22]}"} for i in files]

df_osc_times = pd.read_csv("csv/oscillation_times_remarks.csv")

#%%

date, tree, measurement = lib_streamlit.get_measurement()
tree = tree[-2:]
measurement = measurement[-1]
# tree = f"BK{tree}"
# measurement = f"0{measurement}"

columns = st.columns(3)
with columns[0]:
    acc_axis = st.radio("ACC axis",["x","y","z"], horizontal=True, index=2)
with columns[1]:
    tail = st.radio("Tail length (the length of the zero signal at the start and at the end).",[0,2,4,8,10], horizontal=True)
with columns[2]:
    only_fft = st.radio("Skip signal plot above the FFT (switch to False to see the effect of nonzero tail setting).",[True, False], horizontal=True)
    interactive_fft = st.radio("Allow interactive reading of data from FFT specturm",[True, False], horizontal=True, index=1)

#%%

year,month,day=date.split("-")
df_optics = dt.read_data(f"../data/parquet/{year}_{month}_{day}/BK{tree}_M0{measurement}.parquet")
df_optics = df_optics - df_optics.iloc[0,:]
df_acc = pd.read_csv(f"../data/acc/csv/{date}-BK{tree}-M0{measurement}.csv", index_col=0)
df_elasto = dt.read_data(f"../data/parquet/{year}_{month}_{day}/BK{tree}_M0{measurement}_pulling.parquet")

#%%

"Probes from optics"

options_data = [i for i in df_optics.columns if "Y" in i[1]]
options_data.sort()
options = st.multiselect(
    "Which probes from Optics you want to plot and analyze?",
    options_data,
    [("Pt3","Y0"),("Pt4","Y0")])

#%%

t = np.arange(0, len(df_acc))/100
df_acc.index = t

#%%

row = df_osc_times[
    (df_osc_times['date']==f"{date}") &
    (df_osc_times['tree']==f"BK{tree}") &
    (df_osc_times['measurement']==f"M0{measurement}")
     ]
start, end = row.loc[:,["start","end"]].values.reshape(-1)
f"Default values. Start: {start}, End: {end}"

start = st.number_input("start", value=start)
if not np.isfinite(end):
    end = df_optics.index[-1]
end = st.number_input("end", value=end)
# probe = row['probe'].iat[0]
# if pd.isna(probe):
#     probe = "Pt3"
acc_columns = [i for i in df_acc.columns if "_"+acc_axis in i or "_"+acc_axis.upper() in i]
df = df_acc[acc_columns].abs().idxmax()

release_acc = df[df.sub(df.mean()).div(df.std()).abs().lt(1)].mean()
release_optics = dt.find_release_time_optics(df_optics)
df_acc.index = df_acc.index - release_acc + release_optics

f"""
* Release time is {release_optics} in optics time. 
* Release time in ACC time has been establieshed from the ACC in the following table. However, there is a room for improvements. The release time estableshed
  from ACC should be the time when acceleration jumps from nonzero values.
* The difference between both realease times is {release_acc-release_optics}.
"""
df[df.sub(df.mean()).div(df.std()).abs().lt(1)].T

colnames = ["Full time domain","Release ±1 sec", "Oscillations", "FFT analysis", "Interactive FFT"]

if not interactive_fft :
    colnames = colnames[:-1]
columns = st.columns(len(colnames))
    
for c,text in zip(columns, colnames):
    with c:
        "## "+text

def create_images(df, column=None, start=None, end=None, release_optics=None, tmax = 1e10, only_fft=only_fft):
    ans = []
    for i,limits in enumerate([[0, tmax], [release_optics-1,release_optics+1], [start,end]]): 
        fig, ax = plt.subplots()
        df.loc[limits[0]:limits[1],column].plot(ax=ax,lw=2)
        ax.grid()
        if i < 2:
            # df_optics[(probe,"Y0")].plot()
            ax.axvline(release_optics, color='k', alpha=0.4, lw=0.5, linestyle="--")
            # ax[1].axvline(release_optics, color='k')
        ans += [fig]
    try:
        out = fftdt.do_fft_for_one_column(df.loc[start:end, :], column, create_image=False, preprocessing=lambda x:fftdt.extend_series_with_zeros(x,tail=tail))
        fig = fftdt.create_fft_image(**out, 
                                     only_fft=only_fft, 
                                     ymin = 0.0001)
        ans += [fig]
    except:
        out = None
        ans += [None]
    return ans, out

def uloz(ans, c, date, tree, measurement):
    for i in range(4):
        if ans[i] is None:
            return None
        else:
            ans[i].savefig(f"temp/{i}.png")
    imgs = [Image.open(f"temp/{i}.png") for i in range(4)]
    width, height = imgs[0].size
    cs = {
        "Data1_A01_x":"A01",
        "Data1_A02_x":"A02",
        "Data1_A03_x":"A03",
        "Data1_ACC2_X_axis":"A02",
        "Data1_ACC4_X_axis":"A04",
        "Data1_A01_z":"A01",
        "Data1_A02_z":"A02",
        "Data1_A03_z":"A03",
        "Data1_ACC2_Z_axis":"A02",
        "Data1_ACC4_Z_axis":"A04",
        ('Elasto(90)', 'nan'):"Elasto",
        ('Pt3', 'Y0'):"Pt3",
        ('Pt4', 'Y0'):"Pt4"}
    c_fixed = cs[c]
    nadpis = f"{date} BK{tree} M0{measurement} {c_fixed}"
    font = ImageFont.load_default()    
    # Vytvoření nového obrázku s prostorem pro nadpis
    nadpis_height = 60  # výška prostoru pro nadpis (může se měnit podle velikosti fontu)
    combined_image = Image.new('RGB', (2 * width, 2 * height + nadpis_height), (255, 255, 255))
    
    # Vložení nadpisu
    draw = ImageDraw.Draw(combined_image)
    bbox = draw.textbbox((0, 0), nadpis, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (combined_image.width - text_width) // 2
    text_y = (nadpis_height - text_height) // 2
    draw.text((text_x, text_y), nadpis, fill="black", font=font)
    
    # Vložení obrázků do kombinovaného obrázku pod nadpis
    combined_image.paste(imgs[0], (0, nadpis_height))
    combined_image.paste(imgs[1], (width, nadpis_height))
    combined_image.paste(imgs[2], (0, height + nadpis_height))
    combined_image.paste(imgs[3], (width, height + nadpis_height))
    
    # Uložení výsledného obrázku
    combined_image.save(f"temp/{date}_BK{tree}_M0{measurement}_{c_fixed}.png")    


def nakresli_grafy(out,ans):
    if out is not None:
        f"### {c}, freq = {out['peak_position']:.3f} Hz"
        columns = st.columns(len(colnames))
        for j,a in enumerate(ans):
            with columns[j]:
                st.pyplot(a)
                plt.close(a)
        if not interactive_fft:
            return
        with columns[4]:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=out['xf_r'], y=out['yf_r'], mode='markers'))
            # Nastavení rozsahu osy x, logaritmické osy y a velikosti obrázku
            fig.update_layout(
                xaxis=dict(range=[0, 3], title='freq.'),
                yaxis=dict(type='log', title='FFT'),
                width=400,  # Zmenšení šířky grafu
                height=300,  # Zmenšení výšky grafu
            )
            st.plotly_chart(fig)


for i,c in enumerate(acc_columns):
    ans, out = create_images(df=df_acc, column=c, start=start, end=end, release_optics=release_optics)
    # uloz(ans,c, date, tree, measurement)
    nakresli_grafy(out,ans)

df_optics = df_optics[options]
df_optics = fftdt.interp(df_optics, np.arange(df_optics.index.min(),df_optics.index.max(),0.01))

for i,c in enumerate(options):
    ans, out = create_images(df=df_optics, column=c, start=start, end=end, release_optics=release_optics)
    # uloz(ans,c, date, tree, measurement)
    nakresli_grafy(out,ans)

c = ('Elasto(90)', 'nan')
df_elasto = df_elasto[[c]]
df_elasto = fftdt.interp(df_elasto, np.arange(df_elasto.index.min(),df_elasto.index.max(),0.01))

ans, out = create_images(df=df_elasto, column=c, start=start, end=end, release_optics=release_optics)
# uloz(ans,c, date, tree, measurement)
nakresli_grafy(out, ans)
plt.close('all')
            

"""

## Popis

* Podle nastavení dne, stromu a měření se vykreslí data pro akcelerometry, 
Pt3 a Pt4 z optiky a Elastometr.
* Kreslí se tři časové průběhy pro vizuální kontrolu (všechno, detail vypuštění a oscilatorická část).
* Ve čtvrtém sloupci je buď FFT nebo FFT společně s analyzovaným signálem (pokud se třeba na začátek a na konec přidává nula).
* Je možné přidat pátý sloupec s interkativním FFT grafem pro zobrazení detailu a odečtení hodnot.
* Každý obrázek se dá zvětšit na fullscreen tlačítkem, které se objeví po najetí myší na obrázek. Zpět je ESC nebo ikonka.
* [Videoukázka](https://ctrlv.tv/Xh7H)

"""
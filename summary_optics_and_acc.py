#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:37:18 2024

@author: marik
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import FFT_spectrum as fftdt
import lib_dynatree as dt
from PIL import Image, ImageDraw, ImageFont
import resource

limit_in_gb = 5
limit_in_bytes = limit_in_gb * 1024 * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (limit_in_bytes,limit_in_bytes))

plt.rcParams["figure.figsize"] = (10,6)

fs = 100 # resampled signal
# files = glob.glob("../01_Mereni_Babice_*_optika_zpracovani/csv/*")
# data = [{'year':i[24:28], 'month': i[22:24], 'day': i[20:22], 'measurement': i[-6:-4], 'tree':i[-10:-8],
#          'date': f"{i[24:28]}-{i[22:24]}-{i[20:22]}"} for i in files]

df_osc_times = pd.read_csv("csv/oscillation_times_remarks.csv")

#%%

acc_axis = "z"
tail = 0
only_fft = True

#%%

def main(date,tree, measurement):
    year,month,day=date.split("-")
    df_optics = dt.read_data(f"../01_Mereni_Babice_{day}{month}{year}_optika_zpracovani/csv/BK{tree}_M0{measurement}.csv")
    df_optics = df_optics - df_optics.iloc[0,:]
    df_acc = pd.read_csv(f"../acc/csv/{date}-BK{tree}-M0{measurement}.csv", index_col=0)
    df_elasto = dt.read_data(f"../01_Mereni_Babice_{day}{month}{year}_optika_zpracovani/csv_extended/BK{tree}_M0{measurement}.csv")

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
    
    acc_columns = [i for i in df_acc.columns if "_"+acc_axis in i or "_"+acc_axis.upper() in i]
    df = df_acc[acc_columns].abs().idxmax()
    
    release_acc = df[df.sub(df.mean()).div(df.std()).abs().lt(1)].mean()
    release_optics = dt.find_release_time_optics(df_optics)
    df_acc.index = df_acc.index - release_acc + release_optics
    
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
        out = fftdt.do_fft_for_one_column(df.loc[start:end, :], column, create_image=False, preprocessing=lambda x:fftdt.extend_series_with_zeros(x,tail=tail))
        fig = fftdt.create_fft_image(**out, 
                                     only_fft=only_fft, 
                                     ymin = 0.0001)
        ans += [fig]
        return ans, out
    
    def uloz(ans, c, date, tree, measurement):
        for i in range(4):
            ans[i].savefig(f"temp/{i}.png")
        imgs = [Image.open(f"temp/{i}.png") for i in range(4)]
        width, height = imgs[0].size
        cs = {
            "Data1_A01_z":"A01",
            "Data1_A03_z":"A03",
            'Data1_A02_z':"A02",            
            "Data1_ACC2_Z_axis":"A02",
            "Data1_ACC4_Z_axis":"A04",
            ('Elasto(90)', 'nan'):"Elasto",
            ('Pt3', 'Y0'):"Pt3",
            ('Pt4', 'Y0'):"Pt4"}
        c_fixed = cs[c]
        nadpis = f"{date} BK{tree} M0{measurement} {c_fixed}"
        font = ImageFont.truetype("arial.ttf", 40)
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

    def uloz(ans, c, date, tree, measurement):
        pass
            
    for i,c in enumerate(acc_columns):
        ans, out = create_images(df=df_acc, column=c, start=start, end=end, release_optics=release_optics)
        uloz(ans,c, date, tree, measurement)
        plt.close('all')
        
    options = [("Pt3","Y0"),("Pt4","Y0")]
    
    df_optics = df_optics[options]
    df_optics = fftdt.interp(df_optics, np.arange(df_optics.index.min(),df_optics.index.max(),0.01))
    
    for i,c in enumerate(options):
        ans, out = create_images(df=df_optics, column=c, start=start, end=end, release_optics=release_optics)
        uloz(ans,c, date, tree, measurement)
        plt.close('all')
    
    c = ('Elasto(90)', 'nan')
    df_elasto = df_elasto[[c]]
    df_elasto = fftdt.interp(df_elasto, np.arange(df_elasto.index.min(),df_elasto.index.max(),0.01))
    
    ans, out = create_images(df=df_elasto, column=c, start=start, end=end, release_optics=release_optics)
    uloz(ans,c, date, tree, measurement)
    plt.close('all')
            
df = dt.get_all_measurements()
for index, row in df.iterrows():
    print(row["day"], row["tree"], row["measurement"])
    try:
        main(row["day"], row["tree"], row["measurement"])
    except:
        print(f"Something failed for ", row["day"], row["tree"], row["measurement"])


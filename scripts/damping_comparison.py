# %%
import os
#os.environ["PREFIX_DYNATREE"] = "/home/marik/dynatree/scripts/"
#os.environ["DYNATREE_DATAPATH"] = "/home/marik/dynatree/data/"
import sys
import numpy as np
import logging
import pandas as pd
from dynatree.find_measurements import get_all_measurements, available_measurements
from dynatree.dynatree import DynatreeMeasurement, get_bad_rating, logger
from dynatree.damping import DynatreeDampedSignal
from parallelbar import progress_map
import plotly.graph_objects as go
import config
logger.setLevel(logging.WARNING)

# %%
df = get_all_measurements(method='all').iloc[:,:4]
df = df[df.measurement!="M01"]
df_failed_FFT = pd.read_csv(config.file["FFT_failed"] )
df_failed_stars_elasto = get_bad_rating(key ='max') # mark as failed if all people marked is as failed.
df_failed_stars_elasto = df_failed_stars_elasto[df_failed_FFT.columns]
df_failed = pd.concat([df_failed_stars_elasto, df_failed_FFT], axis = 0).reset_index(drop = True)
#df = df.iloc[:10]

# %%
df_failed_rows = list(df_failed.itertuples(index=False, name=None))
df_failed_rows
# %%
def process_row(row, fig=True):
    data = {}
    # if fig:
    #     # create empty plotly figure
    #     fig = go.Figure()
    # else:
    #     fig = None
    m = DynatreeMeasurement(day=row['date'],
                            tree=row['tree'],
                            measurement=row['measurement'],
                            measurement_type=row['type'])
    for source in ["Pt3","Pt4","Elasto(90)","blueMaj", "yellowMaj"]:
        test = (row['type'],row['date'],row['tree'], row['measurement'], source,)
        if test in df_failed_rows:
            data[*row, source] = [None] *3
            logger.warning(f"Measurement {m}, probe {source} marked as failed, skipping")
            continue
        try:
            s = DynatreeDampedSignal(measurement=m, signal_source=source, #dt=0.0002,
                                    # damped_start_time=54
                                    )
            data[*row, source] = [s.ldd_from_two_amplitudes(max_n=5)[i] for i in ["LDD","R","n"]]
        except Exception as e:
            logger.error(f"Error processing {row['date']} {row['tree']} {row['measurement']} {source}: {e}")
            data[*row, source] = [None] * 3
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reset_index()
    df.columns = ['experiment', 'LDD', 'R', 'n']

    # Ensure data types even if all data are NaN
    required_columns = {
        'experiment': 'object',
        'LDD': 'float64',
        'R': 'float64',
        'n': 'float64'
    }
    df = df.astype(required_columns)

    return {'data':df}

list_m = [i for _,i in df.iterrows()]
ans = progress_map(process_row, list_m)
# %%
ans

# %%

process_row(list_m[0])['data'].dtypes

# %%
ans_data = pd.concat([i['data'] for i in ans if len(i['data'])>0], ignore_index=True)
ans_data

# %%

# Expand the first column of the DataFrame to multiple columns. Keep the rest of the columns as they are.
data = ans_data.experiment.apply(pd.Series)
data = data.rename(columns={0:"day", 1:"tree", 2:"measurement", 3:"type", 4:"source"})
# Now we can concatenate the data with the rest of the columns
data = pd.concat([data, ans_data.drop(columns=['experiment'])], axis=1)  
data
# %%

data.to_csv(config.file['outputs/damping_comparison'], index=False)

# %%

# Take the lines with the same day, tree, measurement and type and find mean and
# std for LDD. 

data_mean = data.groupby(['day', 'tree', 'measurement', 'type']).agg(
    LDD_mean=('LDD', 'mean'),
    LDD_std=('LDD', 'std'),
    R_mean=('R', 'mean'),
    R_std=('R', 'std'),
    n_mean=('n', 'mean'),
    n_std=('n', 'std')
).reset_index()

# %%
data_mean
# %%
data_mean.to_csv(config.file['outputs/damping_comparison_stats'], index=False)
# %%

from dynatree.dynatree import DynatreeMeasurement
from dynatree.damping import DynatreeDampedSignal
m = DynatreeMeasurement(day="2022-08-16", tree="BK08", measurement="M02")
sig = DynatreeDampedSignal(m, signal_source="Elasto(90)")

# %%


# %%
import os
os.environ["PREFIX_DYNATREE"] = "/home/marik/dynatree/scripts/"
os.environ["DYNATREE_DATAPATH"] = "/home/marik/dynatree/data/"
import sys
import numpy as np
import pandas as pd
from dynatree.find_measurements import get_all_measurements, available_measurements
from dynatree.dynatree import DynatreeMeasurement
from dynatree.damping import DynatreeDampedSignal
from parallelbar import progress_map
import plotly.graph_objects as go
import config
# %%

df = get_all_measurements()
df

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
        try:
            s = DynatreeDampedSignal(measurement=m, signal_source=source, #dt=0.0002,
                                    # damped_start_time=54
                                    )
            data[*row, source] = [s.ldd_from_two_amplitudes(max_n=5)[i] for i in ["LDD","R","n"]]
            # if fig:
            #     scaling = np.max(np.abs(s.damped_signal))    
            #     fig.add_trace(go.Scatter(
            #         x=s.damped_time, 
            #         y=s.damped_signal/scaling, 
            #         mode='lines', 
            #         name=source,
            #         line=dict(width=1.5),
            #         hovertemplate=f"{source}: %{{y:.2f}}<extra></extra>"
            #     ))
        except Exception as e:
            print(f"Error processing {row['date']} {row['tree']} {row['measurement']} {source}: {e}")            
            data[*row, source] = [None, None, None]
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df.reset_index()
    df.columns = ['experiment', 'LDD', 'R', 'n']

    return {'data':df, 
    # 'figure': fig
    }



#data = [process_row(row) for i,row in df.iloc.iterrows()]
#data = pd.concat(data, ignore_index=True)
#data

list_m = [i for _,i in df.iterrows() if i['measurement']!="M01"]
ans = map(process_row, list_m)

# %%
ans_data = pd.concat([i['data'] for i in ans], ignore_index=True)
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

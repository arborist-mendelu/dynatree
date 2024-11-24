import pandas as pd

automat = pd.read_csv("./flask_tuky/signal_ratio_upraveny.csv", index_col=0)
new = new.set_index("file")[["liked"]]
new.columns = ["valid"]
new

#%%

data = pd.read_csv("./flask_tuky/FFT_acc_knock.csv", index_col=None)
data["valid"] = True
data['timecoords'] = data['day'].astype(str) + "_" + data['type']
data = data.set_index("filename")

#%%
data.update(new["valid"])

#%%
data = data.reset_index()

#%%
data["manual_peaks"] = None
#%%
data.to_csv("upraveny_pro_import_3.csv")
 #%%
 data.columns

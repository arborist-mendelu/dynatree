import pandas as pd

df_manual_failed = pd.read_csv("./dynatree_summary/FFT_acc_knock_fail_manual.csv",
                               index_col=None, header=None,
                               names=["filename"])
df_manual_failed["valid"] = False
df_manual_failed=df_manual_failed.set_index("filename")
#%%
df_auto = pd.read_csv("./dynatree_summary/FFT_acc_knock_auto.csv", index_col=None)
df_auto["valid"] = True
df_auto = df_auto.set_index("filename")

#%%
df_auto.update(df_manual_failed)
df_auto.to_csv("dynatree_summary/FFT_acc_knock.csv")

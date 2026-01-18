# %%
import json
import pandas as pd

with open("proby_na_zemi.json", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data["comments"])
df = df.drop(columns=["id","rating", "directory"])

mask = df["text"].str.contains("Fix by")
df_fix = df[mask].copy()

def extract_probe(text):
    "From string like 'Fix by Pt11. Minor change.' extract Pt11."
    parts = text.split(" ")
    for part in parts:
        if part.startswith("Pt"):
            part = part.strip(".!,;:")
            return part
    return None

def extract_data(text):
    """
    From string like 'fix_2023-07-17_BK07_M05_normal.png' extract 
    date, tree, measurement, and condition.
    """
    parts = text.split("_")
    if len(parts) < 5:
        return None, None, None, None
    date = parts[1]
    tree = parts[2]
    measurement = parts[3]
    condition = parts[4].split(".")[0]
    return date, tree, measurement, condition

df_fix[["date", "tree", "measurement", "type"]] = df_fix["image"].apply(
    lambda x: pd.Series(extract_data(x))
)
df_fix["probe"] = df_fix["text"].apply(extract_probe)
df_fix = df_fix[["type","date", "tree", "measurement", "probe"]]
df_fix = df_fix.sort_values(by=["type","date","tree","measurement","probe"])

df_fix.to_csv("fixes_optics.csv", index=False)

# %%

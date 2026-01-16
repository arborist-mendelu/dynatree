# %%
import pandas as pd
import os
import glob
import tqdm


df = pd.read_csv("data.txt", header=None, names=["path"])
# df = df[:5]
df
# %%

def get_data(a):
    if "ERC2" in a:
        shift = -1
    else:
        shift = 0
    _ = a.split("/")
    tree = _[8+shift][:2]
    tree_no = _[8+shift][2:]
    measurement = _[12+shift].split("_")[1]
    fps = _[11+shift]
    tree, tree_no, measurement, fps
    snimky_path = "/".join(_[:10+shift])+f"/snimky/{tree}_{tree_no}_{measurement}_{fps}/cam0/"
    snimky_path
    files = glob.glob(os.path.join(snimky_path, "*.bmp"))
    len(files)
    # number of lines of the file "{a}/InputValues.tsv"
    input_values_file = f"{a}/InputValues.tsv"
    with open(input_values_file, "r") as f:
        n_lines = sum(1 for line in f) - 1 
    return len(files), n_lines

# apply get_data to each row in the dataframe and create two new columns "num_images" and "num_input_values_lines"
# df[["num_images", "num_input_values_lines"]] = df["path"].apply(lambda x: pd.Series(get_data(x)))
#df.to_csv("data_with_file_counts.txt", index=False)

# %%

for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    try:
        num_images, num_input_values_lines = get_data(row["path"])
        df.at[i, "num_images"] = num_images
        df.at[i, "num_input_values_lines"] = num_input_values_lines
    except Exception as e:
        print(f"Error processing row {i}: {e}")
        df.at[i, "num_images"] = None
        df.at[i, "num_input_values_lines"] = None

# %%

# find rows where num_images != num_input_values_lines
mismatched_rows = df[df["num_images"] != df["num_input_values_lines"]]
mismatched_rows
# %%
mismatched_rows.to_csv("mismatched_file_counts.txt", index=False)
# %%

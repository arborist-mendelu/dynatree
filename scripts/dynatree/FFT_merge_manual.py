"""
Merge raw FFT output with manual peaks from csv file

"""
import pandas as pd
import config

def merge_with_manual_peaks():
    raw_df =  pd.read_csv(config.file['outputs/FFT_csv_tukey_raw'])
    manual_df = pd.read_csv(config.file['FFT_manual_peaks'])
    manual_df["peak"] = manual_df["peaks"].apply(lambda x: x.strip().split(" ")[0])
    manual_df = manual_df.drop("peaks", axis=1)
    manual_df = manual_df.rename({"measurement_type": "type"}, axis=1)

    new_index = list(raw_df.columns[:-1])
    raw_df = raw_df.set_index(new_index)
    manual_df = manual_df.set_index(new_index)
    manual_df["peak"] = manual_df["peak"].astype(float)

    raw_df.update(manual_df)
    raw_df.to_csv(config.file['outputs/FFT_csv_tukey'])

def main():
    merge_with_manual_peaks()

if __name__ == '__main__':
    main()
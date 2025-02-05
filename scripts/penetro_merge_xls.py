import glob
import pandas as pd
import numpy as np
import config

def penetrologger():
    def fixname(string):
        string = string.split('/')[-1]
        string = string.split('.')[0]
        string = string.replace("penetrologger ","")
        return string
    def fix_column_name(string):
        if not ("Unnamed" in string):
            return string
        string = string.replace("Unnamed: ", "")
        string = int(string)-4
        return string

    def fix_tree_name(t):
        t = f"{t:02d}"
        if t == "18":
            return "JD18"
        return f"BK{t}"

    def fix_date(s):
        s = f"{s}"
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"

    # Funkce, která nahradí nuly na konci řádku za NaN
    def replace_trailing_zeros(row):
        # Zjistí, kde jsou nuly na konci řádku a nahradí je NaN
        for i in range(len(row) - 1, -1, -1):  # Procházíme řádek zpětně
            if row.iloc[i] == 0:
                row.iloc[i] = np.nan  # Nahradíme nulu NaN
            else:
                break  # Jakmile narazíme na hodnotu jinou než nula, zastavíme
        return row

    files = glob.glob(config.file["penetrologgers"])
    data = {fixname(file): pd.read_excel(file) for file in files}

    df = pd.concat(data)
    df = df.reset_index()
    df = df.drop( df.columns[1], axis=1)

    df.columns.values[0] = 'day'
    df.columns.values[4] = 'tree'

    df["day"] = df["day"].apply(fix_date)
    df['tree'] = df['tree'].apply(fix_tree_name)

    df.columns = [fix_column_name(i) for i in df.columns]
    df = df.apply(replace_trailing_zeros, axis=1)

    return (df)

if __name__ == "__main__":
    df = penetrologger()
    df.to_csv(config.file["penetrologger.csv"], index=False)
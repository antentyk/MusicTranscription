import pandas as pd

def time_series_from_txt(filename):
    return pd.read_csv(filename, sep="\t")

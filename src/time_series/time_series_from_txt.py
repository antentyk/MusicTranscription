import pandas as pd

def time_series_from_txt(filename):
    """
    Returns time series representation of the song (Onset, Offset, MidiPitch)
    following the structure of .txt representation of the song
    in MAPS database

    Args:
        filename (str): path to the file to be processed
    
    Returns:
        pandas.Dataframe: time series song representation
    """
    return pd.read_csv(filename, sep="\t")

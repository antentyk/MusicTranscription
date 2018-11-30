import pandas as pd


def time_series_from_txt(filename):
    return pd.read_csv(filename, sep="\t")


if __name__ == "__main__":
    pass
    # import os
    # import sys
    #
    # sys.path.append(os.path.abspath("../"))
    #
    # from config import config
    # from log import log
    #
    # print()
    #
    # filename = config["path_to_MAPS"] + "MAPS_AkPnBcht_2/AkPnBcht/MUS/MAPS_MUS-alb_se3_AkPnBcht.txt"
    #
    # df = time_series_from_txt(filename)

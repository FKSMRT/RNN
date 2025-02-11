import numpy as np
import pandas as pd
import os
import csv
import glob
from natsort import natsorted


# 後半半分のデータを抽出
def get_after(dirpath,filename,savepath):
    """
    指定したCSVファイルの後半半分のデータを抽出し、新たなCSVファイルとして保存する。

    [パラメータ]
    dirpath : CSVファイルが存在するディレクトリのパスを指定
    filename : CSVファイルを指定
    """
    csvfile = (glob.glob(os.path.join(dirpath , filename)))
    csvfile = natsorted(csvfile)
    print(csvfile)

    for i in range(len(csvfile)):
        df = pd.read_csv(csvfile[i],header=None,usecols=[1],skiprows=8)
        start = int(len(df) / 2)
        df = df.iloc[start:,:]

        savefile = os.path.join(savepath , "LF_ch1_" + str(i+1) + ".csv")
        print(savefile)
        df.to_csv(savefile,header=False,index=False)
    

dir = os.path.expanduser("~")
dirpath = os.path.join(dir + "\Python\B4_1\Fragrance_Lavender\LF")
savepath = os.path.join(dir + "\Python\B4_1\Fragrance_Lavender\BPF\LF_ch1_after")
get_after(dirpath,"*.csv",savepath)


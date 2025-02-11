import numpy as np
import pandas as pd
import os
import csv
import glob
from natsort import natsorted
from scipy import signal

def bpf(wave, fs, fe1, fe2, n):
    """
    [パラメータ]
    信号
    サンプリング周波数
    カットオフ周波数１
    カットオフ周波数２
    フィルタの次元
    """
    nyq = fs / 2.0
    b, a = signal.butter(9, [fe1/nyq, fe2/nyq],btype = "band")
    for i in range(0,n):
        wave =signal.filtfilt(b,a,wave)
    
    return wave

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

        wave = df.to_numpy()
        print(wave)
        fs = 1000
        fe1 = 15
        fe2 = 150
        n = len(df)
        data = bpf(wave,fs,fe1,fe2,n)

        df1 = pd.Series(data)
        start1 = int(len(df1) / 2)
        df1 = df1.iloc[start1:start1 + 1048576,:]

        savefile = os.path.join(savepath , "LF_ch1_" + str(i+1) + ".csv")
        print(savefile)
        df1.to_csv(savefile,header=False,index=False)
    

dir = os.path.expanduser("~")
dirpath = os.path.join(dir + "\Python\B4_1\Fragrance_Lavender\BPF\LF_ch1_after(size_org)")
savepath = os.path.join(dir + "\Python\B4_1\Fragrance_Lavender\BPF")
get_after(dirpath,"LF_ch1_1.csv",savepath)


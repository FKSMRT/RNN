import numpy as np
import pandas as pd
import os
import csv
import glob
from natsort import natsorted
from scipy import signal


# データの読み込み method_1
def DataReader(dirpath,filename):

    """
    指定ディレクトリから指定ファイルを読み込む

    [パラメータ]
    dirpath : 読み込むファイルが存在するフォルダのパスを指定
    filename : 読み込むファイルの名前を指定

    戻り値：list型
    """

    csvfile = (glob.glob(os.path.join(dirpath , filename)))
    csvfile = natsorted(csvfile)
    print("csvfile" + "\n", csvfile)

    dataset = []
    for i in range(len(csvfile)):
        data = np.loadtxt(csvfile[i],dtype = int,delimiter = ".")
        data = data.tolist()
        print("data" + "\n", data[0:10], type(data))
        dataset += [data[0:1048576]]
        print("dataset" + "\n", len(dataset),len(dataset[i]))
    dataset = np.array(dataset)
    #print("data.shape" + "\n", data.shape)
    print("dataset.shape" + "\n", dataset.shape)

    return dataset


def sort(data):
    dataset = []
    print("org_data" + "\n", data)
    for i in range(3):
        for j in range(8):
            dataset += [data[i + (3*j)]]
    print("sort_data" + "\n", dataset)
    return dataset


def dawnsampling(data,dawn_Hz = 1000):
    data = np.array(data)
    dawn_dataset = []
    print(data.shape)
    print(data.shape[0])
    print(data.shape[1])

    for i in range(data.shape[0]):
        list = []
        for j in range(data.shape[1]):
            if j * dawn_Hz > data.shape[1]:
                break
            elif j == 0:
                list += [data[i][j]]
            else :
                list += [data[i][(j * dawn_Hz)-1]]
        dawn_dataset += [list]
    dawn_dataset = np.array(dawn_dataset, dtype = np.uint32)

    return dawn_dataset


def standardization(data):
    data_std_list =[]
    for i in range(0,len(data)):
        data_mean = np.mean(data[i])
        data_std = np.std(data[i],ddof=1)
        std_list =[]
        for j in range(0,len(data[i])):
            data_std = (data[i][j] - data_mean) / data_std
            std_list += [data_std]
        data_std_list += [std_list]
    
    return data_std_list


def savedata(data,path):
    for i in range(len(data)):
        savepath = os.path.join(path , "LF_ch1_" + str(i+1) + "(size_1049).csv")
        np.savetxt(savepath, data[i], delimiter=",", fmt = "%f")

    

dir = os.path.expanduser("~")
dirpath = os.path.join(dir + "\Python\B4_1\Fragrance_Lavender\LF\LF_ch1_after(size_org)")
savepath = os.path.join(dir + "\Python\B4_1\Fragrance_Lavender\LF\LF_ch1_after(size_1049)")

# CSVファイルをダウンロード
load_dataset = DataReader(dirpath,"*.csv")

load_dataset = sort(load_dataset)

# ダウンサンプリングを行う
dataset = dawnsampling(load_dataset)
print(dataset.shape)

# 標準化を行う
#dataset = standardization(dawn_dataset)

# CSVファイルとして保存
savedata(dataset,savepath)

"""
バニラ香料実験で得られたデータを分割するプログラム。
用いるデータはバンドパスフィルタ（BPF）により前処理されたもである。

・データ説明
サンプリング　1kHz
ch1 ペースメーカー付近
CH2 幽門付近
A:control(1)
B:溶媒のみ(2)
C:低濃度(3)
D:高濃度(4)
BPFフォルダ内：分割後データ
ファイル被験者	試料
1.csv	1	A前半
2.csv	2	A前半
3.csv	3	A前半
4.csv	4	A前半
5.csv	5	A前半
6.csv	6	A前半
7.csv	7	A前半
8.csv	8	A前半
9.csv	1	A後半
10.csv	2	A後半
11.csv	3	A後半
12.csv	4	A後半
13.csv	5	A後半
14.csv	6	A後半
15.csv	7	A後半
16.csv	8	A後半
17.csv	1	B前半
18.csv	2	B前半
19.csv	3	B前半
20.csv	4	B前半
21.csv	5	B前半
22.csv	6	B前半
23.csv	7	B前半
24.csv	8	B前半
25.csv	1	B後半
26.csv	2	B後半
27.csv	3	B後半
28.csv	4	B後半
29.csv	5	B後半
30.csv	6	B後半
31.csv	7	B後半
32.csv	8	B後半
33.csv	1	C前半
34.csv	2	C前半
35.csv	3	C前半
36.csv	4	C前半
37.csv	5	C前半
38.csv	6	C前半
39.csv	7	C前半
40.csv	8	C前半
41.csv	1	C後半
42.csv	2	C後半
43.csv	3	C後半
44.csv	4	C後半
45.csv	5	C後半
46.csv	6	C後半
47.csv	7	C後半
48.csv	8	C後半
49.csv	1	D前半
50.csv	2	D前半
51.csv	3	D前半
52.csv	4	D前半
53.csv	5	D前半
54.csv	6	D前半
55.csv	7	D前半
56.csv	8	D前半
57.csv	1	D後半
58.csv	2	D後半
59.csv	3	D後半
60.csv	4	D後半
61.csv	5	D後半
62.csv	6	D後半
63.csv	7	D後半
64.csv	8	D後半
"""

import os
import csv
import glob
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from natsort import natsorted


# データの読み込み method_1
def DataReader(dirpath,filename):

    """
    Python/B4_1/AutoEncoder/VanillaFragrance/BPF/BPF_ch1_*.csv　を読み込む。

    * 1～64の計64ファイルが存在。番号については上記説明参照。

    [パラメータ]
    dirpath : 読み込むファイルが存在するフォルダのパスを指定
    filename : 読み込むファイルの名前を指定
    """
    dataset = []

    csvfile = (glob.glob(os.path.join(dirpath + filename)))
    csvfile = natsorted(csvfile)

    for i in range(len(csvfile)):
        data = []
        with open(csvfile[i], "r", encoding = "utf-8") as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                data += line
        data = list(map(float, data))
        dataset += [data]
    return dataset


# データの正規化 method_2
def data_normalization(data) :

    """
    データを -1 ~ 1 で正規化を行う。
    分割した後のデータを正規化するため、この関数では2次元配列に対する処理を前提に考えている。

    「パラメータ」
    data : 正規化を行うデータを指定
    """
    data_norm_list = []
    for i in range(0,len(data)):
        data_min = min(data[i])
        data_max = max(data[i])
        norm_list = []
        for j in range(0,len(data[i])):
            if data_max - data_min == 0 :
                data_norm = 0
                norm_list += [data_norm]
            else :
                data_norm = (data[i][j] - data_min) / (data_max - data_min)
                data_norm = (data_norm * 2) -1
                norm_list += [data_norm]
        
        data_norm_list += [norm_list]

    return data_norm_list


# データの分割 method_3
def split_list(data,num):

    """
    読み込んだCSVファイルのデータを分割する。
    指定したdataのリストをnum秒ごとの要素に分割する。
    データ数を増やすために1秒ごとずらしてデータを搾取する。
    つまり、データ数は(num[s] x 1040)になる

    [パラメータ]
    data : 分割を行いたいデータの指定
    num : 分割したい1つのデータの秒数を指定（セル数）
    """

    list_data = []
    for i in range(0,len(data)):
        if i+num > 1049:
            break
        list_data += [data[i : i + num]]
    list_data = np.array(list_data)
    return list_data


# 正解ラベルを1次元に変換する method_4
def label_1D(data):
    """
    ラベルデータ作成の際に、間違って1つの値ごと(1セルずつ）にラベルを付与してしまった。
    そのため、ここで指定秒数に区切った後のデータに対してラベル付けを再度行う。
    このメソッドでは、ndarray型の正解ラベルデータを1次元化する

    [パラメータ]
    data : 1次元化したい正解ラベルを指定
    """

    label = []
    for idx in range(0,data.shape[0]):
        label += [int(data[idx][0])]
    label = np.array(label)
    return label


### 以下 実行プログラム ###

dir = os.path.expanduser("~")
dirpath = os.path.join(dir + "\Python\B4_1\AutoEncoder\VanillaFragrance\BPF")
savepath = os.path.join(dir + "\Python\B4_1\RNN\VF_data")

# CSＶファイルをダウンロード
VF_dataset = DataReader(dirpath,"\BPF_ch1_after\BPF_ch1_*.csv") # DataReader()を用いてCSVファイルを読み込む
VF_dataset = np.array(VF_dataset) # ndarray 化

VF_label = DataReader(dirpath,"\Correct_label\*.csv") # DataReader()を用いてCSVファイルを読み込む
VF_label = np.array(VF_label) # ndarray 化


# 正解ラベルを1次元化する
y = label_1D(VF_label)
print(y)
print(y.shape)


# データの正規化
VF_normalization_dataset = data_normalization(VF_dataset) # data_normalization()を用いてデータを正規化
VF_normalization_dataset = np.array(VF_normalization_dataset) # ndarray 化


# データの分割
split_num = 20

VF_split_dataset = [] # 分割リストの用意
for n in range(len(VF_normalization_dataset)): # 読み込んだ32のデータセットを20秒ごとに分割する
    VF_split_data = split_list(VF_normalization_dataset[n],split_num) 
    VF_split_dataset += [VF_split_data]
x = np.array(VF_split_dataset) # ndarray 化
print(x.shape)


# テストデータの一部データ波形を確認する
fig = plt.figure(figsize = (32,4))
ax = fig.add_subplot(2,10,1)
ax.plot(x[0][1])
plt.show()


# npyファイルとして保存(x)
x_dirpath = os.path.join(savepath + "\VF_x.npy") # npyファイルの保存先パスを指定
if os.path.exists(x_dirpath): # 指定フォルダ内に同じ名前のnpyファイルが存在しているかを確認
    os.remove(x_dirpath) # 同ファイルが存在した場合,npyファイルを削除
np.save(x_dirpath, x)

# npyファイルとして保存(y)
y_dirpath = os.path.join(savepath + "\VF_y.npy") # npyファイルの保存先パスを指定
if os.path.exists(y_dirpath): # 指定フォルダ内に同じ名前のnpyファイルが存在しているかを確認
    os.remove(y_dirpath) # 同ファイルが存在した場合,npyファイルを削除
np.save(y_dirpath, y)
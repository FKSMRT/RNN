"""
バニラ香料実験で得られたデータを分割するプログラム。
用いるデータはバンドパスフィルタ（BPF）により前処理されたもである。

・データ説明
サンプリング　1kHz
ch1 ペースメーカー付近
CH2 幽門付近
A:control(0)
B:溶媒のみ(1)
C:低濃度(2)
D:高濃度(3)
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
    指定ディレクトリから指定ファイルを読み込む

    [パラメータ]
    dirpath : 読み込むファイルが存在するフォルダのパスを指定
    filename : 読み込むファイルの名前を指定

    戻り値：list型
    """
    dataset = []

    csvfile = (glob.glob(os.path.join(dirpath + filename)))
    csvfile = natsorted(csvfile)
    print(csvfile)

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
    データを -1 ~ 1 で正規化を行う (# をコメントアウトすると 0 ~ 1 に正規化)

    「パラメータ」
    data : 正規化を行うデータを指定

    戻り値：list型 
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
                #data_norm = (data_norm * 2) -1 #
                norm_list += [data_norm]
        
        data_norm_list += [norm_list]

    return data_norm_list


# データの分割 method_3
def split_list(data,num):

    """
    指定したdataをnum秒ごとの要素に分割する。

    [パラメータ]
    data : 分割したいデータの指定
    num : 分割したい1つのデータの秒数を指定（セル数）

    入力値：ndarray型
    戻り値：ndarray型
    """

    dataset = []
    for i in range(data.shape[0]):
        x1 = []
        for j in range(data.shape[1] - num):
            x2 = []
            x2 += [data[i][j : j + num]]
            x1 += x2
        dataset += [x1]
    dataset = np.array(dataset)
    return dataset


# 正解ラベルを1次元に変換する method_4
def label_1D(data):

    """
    ラベルデータ作成の際に、間違って1つの値ごと(1セルずつ）にラベルを付与してしまった。
    そのため、ここで指定秒数に区切った後のデータに対してラベル付けを再度行う。
    このメソッドでは、ndarray型の正解ラベルデータを1次元化する

    [パラメータ]
    data : 1次元化したい正解ラベルを指定

    入力値：ndarray型
    戻り値：ndarray型
    """

    label = []
    for idx in range(0,data.shape[0]):
        label += [data[idx][0]]
    label = np.array(label, dtype = np.int16)
    return label


# 正解ラベルを2次元に変換する method_5
def label_2D(data):

    """
    ラベルデータ作成の際に、間違って1つの値ごと(1セルずつ）にラベルを付与してしまった。
    そのため、ここで指定秒数に区切った後のデータに対してラベル付けを再度行う。
    このメソッドでは、ndarray型の正解ラベルデータを2次元化する

    [パラメータ]
    data : 2次元化したい正解ラベルを指定
    入力値：ndarray型
    戻り値：ndarray型
    """

    labelset = []
    for idx_1 in range(0,data.shape[0]):
        label = []
        for idx_2 in range(0,data.shape[1]):
            label += [data[idx_1][idx_2][0]]
        labelset += [label]
    labelset = np.array(labelset, dtype = np.int16)
    return labelset


# 被験者ごとに並べ替える method_6
def sort_subjects(data,label,subjects_num,label_num):

    """
    被験者ごとにまとめたデータセットに変換する
    このメソッドでは、list型のdata,labelを入力する

    [パラメータ]
    data : 並び替えるデータを指定
    label : 並び替えるラベルを指定（dataに対応するラベル）
    subjects_num : 被験者の数
    label_num : ラベルの個数（今回は4クラス）

    戻り値：list型
    """

    sort_idx = []
    sort_idx_A = np.arange(0,subjects_num * label_num , subjects_num)
    sort_idx_A.tolist()

    for i in range(subjects_num) :
        sort_idx_X = sort_idx_A + i
        sort_idx.extend(sort_idx_X)

    sort_data =[]
    sort_label = []
    for j in sort_idx :
        sort_data += [data[j]]
        sort_label += [label[j]]

    return sort_data,sort_label


# データセットから指定の区間を抽出 method_7
def get_data(data,time,start_time = 0):

    """
    データセットから指定の区間を抽出するメソッド
    計測データの 〇分 ~ 〇分 までを新たなデータセットとして取得する
    [パラメータ]
    data : 抽出元のデータセットを指定
    time : 何分間のデータを取得するか指定
    start : 何分からのデータを取得するか指定

    入力値：list型
    戻り値：list型
    """

    min_cell = 52.45 # 1分当たりのセルの個数
    get_time = int(min_cell * time)
    start = int(min_cell * start_time)
    data = np.array(data)
    list = []
    for i in range(data.shape[0]):
        list += [data[i][start:(get_time + start - 1)]]

    return list
 

### 以下 実行プログラム ###

"""---------------初期値を設定---------------"""
time = 10
subjects_num = 8
label_num = 4
split_num = 32
"""-----------------------------------------"""

dir = os.path.expanduser("~")
dirpath = os.path.join(dir + "\Python\B4_1\Fragrance_Vanilla\BPF")
savepath = os.path.join(dir + "\Python\B4_1\RNN\VF_data")

# CSＶファイルをダウンロード
load_dataset = DataReader(dirpath,"\BPF_ch1_after\*.csv") # DataReader()を用いてCSVファイルを読み込む
#load_dataset = DataReader(dirpath,"\BPF_ch1_after_notcontrol\*.csv")
load_label = DataReader(dirpath,"\Correct_label\*.csv") # DataReader()を用いてCSVファイルを読み込む

"""
# 0~10分の区間を抽出
load_dataset = get_data(load_dataset, time)
load_label = get_data(load_label, time)
load_dataset = np.array(load_dataset)
print(load_dataset.shape)
"""

# 被験者ごとに並び変える
(VF_dataset, VF_label) = sort_subjects(load_dataset, load_label, subjects_num, label_num) 
VF_dataset = np.array(VF_dataset) # ndarray 化
VF_label = np.array(VF_label) # ndarray 化
print(VF_dataset.shape)
print(VF_label.shape)

# 正解ラベルを1次元化する
y = label_1D(VF_label)
print(y.shape)

# データの正規化
VF_normalization_dataset = data_normalization(VF_dataset) # data_normalization()を用いてデータを正規化
VF_normalization_dataset = np.array(VF_normalization_dataset) # ndarray 化
print(VF_normalization_dataset.shape)

# データの分割
VF_split_dataset = split_list(VF_normalization_dataset,split_num) 
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

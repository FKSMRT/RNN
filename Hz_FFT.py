import fft_function
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import csv
import glob
from natsort import natsorted

# データの取得
filename = "1.csv"
dir = os.path.expanduser("~")
dirpath = os.path.join(dir + "\Python\B4_1\Fragrance_Lavender\LF\LF_ch_1")
csvfile = (glob.glob(os.path.join(dirpath , filename)))
csvfile = natsorted(csvfile)
print(csvfile)

data = []
for i in range(len(csvfile)):
    df = pd.read_csv(csvfile[i],header=None,usecols=[1],skiprows=8)
    wave = df.to_numpy()
    print("wave : ", wave)
    wave = np.squeeze(wave)
    print("wave : ", wave)
    data += [wave]
    print("data : ", data)

###
#data = data[0]
###
samplerate = 25600
x = np.arange(0,12800)/samplerate
print("x : ", x)

Fs = 4096       #フレームサイズ
overlap = 50    #オーバーラップ率
 
#作成した関数を実行：オーバーラップ抽出された時間波形配列
time_array, N_ave = fft_function.ov(data, samplerate, Fs, overlap)
 
#作成した関数を実行：ハニング窓関数をかける
time_array, acf = fft_function.hanning(time_array, Fs, N_ave)
 
#作成した関数を実行：FFTをかける
fft_array, fft_mean, fft_axis = fft_function.fft_ave(time_array, samplerate, Fs, N_ave, acf)
 
t = np.arange(0, Fs)/samplerate     #グラフ描画のためのフレーム時間軸作成
 
#ここからグラフ描画
# フォントの種類とサイズを設定する。
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Times New Roman'
 
# 目盛を内側にする。
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
 
# グラフの上下左右に目盛線を付ける。
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.yaxis.set_ticks_position('both')
ax1.xaxis.set_ticks_position('both')
ax2 = fig.add_subplot(212)
ax2.yaxis.set_ticks_position('both')
ax2.xaxis.set_ticks_position('both')
 
# 軸のラベルを設定する。
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Signal [V]')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Signal [V]')
 
#データの範囲と刻み目盛を明示する。
ax1.set_xticks(np.arange(0, 2, 0.04))
ax1.set_yticks(np.arange(-5, 5, 1))
ax1.set_xlim(0, 0.16)
ax1.set_ylim(-3, 3)
ax2.set_xticks(np.arange(0, samplerate, 50))
ax2.set_yticks(np.arange(0, 3, 0.5))
ax2.set_xlim(0,200)
ax2.set_ylim(0, 1)
 
# データプロットの準備とともに、ラベルと線の太さ、凡例の設置を行う。
for i in range(N_ave):
    ax1.plot(t, time_array[i], label='signal', lw=1)
 
ax2.plot(fft_axis, fft_mean, label='signal', lw=1)
 
fig.tight_layout()
 
# グラフを表示する。
plt.show()
plt.close()
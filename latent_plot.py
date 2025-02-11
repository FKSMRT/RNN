import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D

def train_val_test_split(x, y, val_subject, test_subject, subjects,label_num):

    """
    訓練データ、テストデータ、評価データ分割メソッド

    [パラメータ]
    x : 入力データ
    y : 入力ラベル
    val_subject : 評価データにあてる被験者の数(list型)
    test_subject : テストデータにあてる被験者(list型)
    subjects = 被験者のリスト(list型)
    label_num : ラベルの数
    """

    subject_idx_1 = [subjects.index(i) for i in test_subject]
    x_test = []
    y_test = []
    for i in subject_idx_1 :
        start1 = label_num * i
        stop1 = start1 + label_num
        x_test += [x[start1:stop1,:,:]]
        y_test += [y[start1:stop1]]
        del subjects[i]
        x_org = np.delete(x,slice(start1,stop1),axis=0)
        y_org = np.delete(y,slice(start1,stop1),axis=0)

    subject_idx_2 = [subjects.index(i) for i in val_subject]
    x_val = []
    y_val = []
    for j in subject_idx_2 :
        start2 = label_num * j
        stop2 = start2 + label_num
        x_val += [x[start2:stop2,:,:]]
        y_val += [y[start2:stop2]]
        del subjects[j]
        x_train = np.delete(x_org,slice(start2,stop2),axis=0)
        y_train = np.delete(y_org,slice(start2,stop2),axis=0)

    x_test = np.squeeze(x_test,axis=0)
    y_test = np.squeeze(y_test,axis=0)
    x_val = np.squeeze(x_val,axis=0)
    y_val = np.squeeze(y_val,axis=0)

    return x_train,y_train,x_val,y_val,x_test,y_test

# 生成波形をGW_fig_num個出力、保存
def GW_output(x_test,GW,GW_fig_num,GW_fig_size,test_subject):

    """
    テストデータとテストデータをもとに予測した生成波形(Generated_waveform)
    を出力するメソッド

    [パラメータ]
    x_test : テストデータを指定
    GW : 予測した生成波形(Generated_waveform)を指定
    GW_fig_num : 出力する画像の数(総数ではなく x_test と GW それぞれの数)
    GW_fig_size : 出力する画像のfigsize
    test_subject : テストデータにあてる被験者（list型)
    """
    num = int(GW_fig_num/5)
    idx_num1 = np.arange(0, x_test.shape[0])
    idx_num2 = np.arange(0, x_test.shape[1])

    fig = plt.figure(figsize=GW_fig_size)
    fig.suptitle("Generated waveform (subject = " + str(test_subject[:]) + ")", fontsize=15)

    for i in range(GW_fig_num):
        idx_1 = random.randint(0,len(idx_num1)-1)
        idx_2 = random.randint(0,len(idx_num2)-1)
        idx_1 = idx_num1[idx_1]
        idx_2 = idx_num2[idx_2]
        # テスト画像を表示
        ax = fig.add_subplot(num, 10, i+1,title=("Test idx:(" + str(idx_1) + " , " + str(idx_2) + ")"))
        ax.plot(x_test[idx_1][idx_2])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # 変換された画像を表示
        ax = fig.add_subplot(num, 10, i+1+GW_fig_num,title=("GW idx:(" + str(idx_1) + " , " + str(idx_2) + ")"))
        ax.plot(GW[idx_1][idx_2])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


# モデルスコアを棒グラフで出力
def score_bar(score,labels,test_subject):

    """
    モデルのスコアをラベルごとに棒グラフとして出力するメソッド

    [パラメータ]
    score : 出力させたいScoreを指定（list型）
    labels : スコアに対応するラベルを指定（list型）
    test_subject : テストデータにあてる被験者（list型)
    """

    x = len(score)
    width = 0.4
    fig =  plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_ylim([0,100])
    ax.bar(labels, score, width, align ="center")
    for i in range(x):
        ax.text(i, score[i], str(round(score[i],1)) + "(%)", horizontalalignment="center")
    ax.set_title("Generated waveform Score(%)")
    plt.title("Score (subject = " + str(test_subject[:]) + ")", fontsize = 15)
    plt.show()



# 2次元で潜在変数をプロット
def plot_results_2D(latent,
                 y_test,
                 test_label,
                 test_subject):

    print(latent.shape)
    print(latent[0])
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1,1,1)
    cmap=cm.tab10
    for i in y_test:
        ax.scatter(latent[i, :, 0],
                        latent[i, :, 1],
                        c = cmap(i),
                        label = test_label[i])
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.legend()
    plt.title("Latent Variables 2D (subject = " + str(test_subject[:]) + ")", fontsize = 15)
    plt.show()


# 3次元で潜在変数をプロット
def plot_results_3D(latent,
                 y_test,
                 test_label,
                 test_subject):

    fig = plt.figure(figsize=(12,10))
    ax = Axes3D(fig)
    cmap=cm.tab10
    for i in y_test:
        ax.scatter(latent[i, :, 0],
                    latent[i, :, 1],
                    latent[i, :, 2],
                    c = cmap(i),
                    label = test_label[i])
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_zlabel("z[2]")
    ax.set_title("Latent Variables 3D (subject = " + str(test_subject[:]) + ")", fontsize = 15)
    ax.legend()
    plt.show()


#　複数データの重心位置を求める




subjects = ["A", "B", "C", "D", "E", "F", "G", "H"]
test_label = ["Control","Only Solvent", "Low Concentration", "High Concentration"]
subjects_num = len(subjects)
label_num = len(test_label)
val_subject = ["H"]
val_num = len(val_subject)
test_subject = ["E"]
test_num = len(test_subject)
GW_fig_num = 10
GW_fig_size = (60,6)

dir = os.path.expanduser("~")
dirpath = os.path.join(dir + "\Python\B4_1\RNN\VF_result")
x_dirpath = os.path.join(dir + "\Python\B4_1\RNN\VF_data\VF_x.npy")
y_dirpath = os.path.join(dir + "\Python\B4_1\RNN\VF_data\VF_y.npy")

x = np.load(x_dirpath)
y = np.load(y_dirpath)
x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x,y,val_subject,test_subject,subjects,label_num)

latent = os.path.join(dirpath + "\latent\latent_['E'].npy")
RMSE = os.path.join(dirpath + "\RMSE\RMSE_['E'].npy")
score = os.path.join(dirpath + "\score\score_['E'].npy")


"""
B_dirpath = os.path.join(dirpath + "\latent_['B'].npy")
C_dirpath = os.path.join(dirpath + "\latent_['C'].npy")
D_dirpath = os.path.join(dirpath + "\latent_['D'].npy")
E_dirpath = os.path.join(dirpath + "\latent_['E'].npy")
F_dirpath = os.path.join(dirpath + "\latent_['F'].npy")
G_dirpath = os.path.join(dirpath + "\latent_['G'].npy")
H_dirpath = os.path.join(dirpath + "\latent_['H'].npy")
"""

latent = np.load(latent)
RMSE = np.load(RMSE)
score = np.load(score)

"""
latentB = np.load(B_dirpath)
latentC = np.load(C_dirpath)
latentD = np.load(D_dirpath)
latentE = np.load(E_dirpath)
latentF = np.load(F_dirpath)
latentG = np.load(G_dirpath)
latentH = np.load(H_dirpath)
"""

print("\n")
print("RMSE \n " + str(RMSE))
print("Score (%) \n" + str(np.mean(score)) )
print("\n")

plot_results_2D(latent,y_test,test_label,test_subject)
plot_results_3D(latent,y_test,test_label,test_subject)

import os
import numpy as np
import pandas as pd
from math import dist

# 重心を算出
def barycenter(latent):
    """
    [パラメータ]
    latent : 潜在変数（3次元サイズ、shape=[香料濃度、データ長、潜在変数の値]）
    """
    x_y_z_sum = np.sum(latent,axis=1)
    barycenter = x_y_z_sum / latent.shape[1]

    return barycenter


# 重心間の距離を算出
def euc_dis(G,num):
    for i in range(len(G)):
        if i == 0:
            x = dist(G[num],G[i])
            EucDis = np.array([x])
        else :
            x = dist(G[num],G[i])
            EucDis = np.append(EucDis,x)
    
    return EucDis


# 重心間の距離算出から表作成
def table_EucDis(G):
    label = ["Only Solvent", "Low Concentration", "High Concentration"]
    a = np.array([])
    num = len(label)
    for i in range(num):
        list = euc_dis(G,i)
        a = np.append(a,list)
    list = np.reshape(a,(num,num))
    df = pd.DataFrame(data=list,index=label,columns=label)
    
    return df

def G_out(subject):
    dir = os.path.expanduser("~")
    path = "Python\B4_1\RNN\LF_result"
    latentpath = "latent"
    latentfile = "latent_['" + subject + "'].npy"
    dirpath = os.path.join(dir , path, latentpath, latentfile)
    latent = np.load(dirpath)
    G = barycenter(latent)
    df = table_EucDis(G)

    return G, df


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

subjects = ["A", "B", "C", "D", "E", "F", "G", "H"]

dir = os.path.expanduser("~")
savepath = os.path.join(dir, "Python\B4_1\RNN\\to_csv")

G = []

for i in subjects:
    print("subject = ", i)
    _G , df = G_out(i)
    print(df)
    G += [_G]
print(G)
G = np.array(G)
G = np.reshape(G, (-1,3))
print(G.shape)

if not os.path.exists(savepath):
    os.mkdir(savepath)

np.savetxt(savepath + "\\G_LF.csv", G, delimiter=",")

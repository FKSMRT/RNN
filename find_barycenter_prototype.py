import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.patches as mpatches
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
def table_EucDis(G,label): 
    a = np.array([])
    for i in range(4):
        list = euc_dis(G,i)
        a = np.append(a,list)
    list = np.reshape(a,(4,4))
    df = pd.DataFrame(data=list,index=label,columns=label)
    
    return df


# データの正規化
def data_normalization(latent,axis=None) :

    """
    データを 0 ~ 1 で正規化を行う 

    「パラメータ」
    data : 正規化を行うデータを指定

    戻り値 : list型 
    """
    mean = latent.mean(axis=axis, keepdims=True)
    std = np.std(latent,axis=axis,keepdims=True)
    result = (latent-mean)/std
    return result


# 表を図として出力する
def df_plot(df):
    plt.rcParams["font.family"] = "Times New Roman"

    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(1,1,1)
    color = np.full_like(df, "", dtype=object)
    for i in range(len(df)):
        for j in range(df.shape[1]):
            if df.iloc[i, j] == 0:
                color[i, j] = 'black'
            else:
                color[i, j] = 'white'
    table = ax.table(cellText=np.round(df.values,3),cellLoc="center",
            colWidths=[0.2]*4,
            rowLabels=df.index,rowLoc="center",rowColours=["silver"]*4,
            colLabels=df.columns,colLoc="center",colColours=["silver"]*4,
            loc="center",
            cellColours=color)
    #fig.tight_layout()
    ax.axis("off")
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.scale(1,2)
    #plt.show()

    return fig


# 一つの香料について重心とデータを3次元プロット
def plot3D(latent,barycenter,Fc):
    """
    [パラメータ]
    latent : 潜在変数（香料指定、2次元サイズ）
    barycenter : 重心（香料指定、2次元サイズ）
    Fc : 香料濃度のラベル（文字列）
    """
    fig = plt.figure(figsize = (12,10))
    ax = Axes3D(fig)
    ax.scatter(latent[:,0],latent[:,1],latent[:,2],s=0.5,c="b",label="Latent of "+Fc)
    ax.scatter(barycenter[0],barycenter[1],barycenter[2],marker="o",s=100,c="r",label="Barycenter of "+Fc)
    ax.legend(fontsize=18)
    ax.tick_params(labelsize=18)
    #plt.show()

    return fig


# すべての香料について重心と潜在変数を3次元プロット
def plot_all(latent,barycenter,label):
    """
    [パラメータ]
    latent : 潜在変数（香料指定、3次元サイズ）
    barycenter : 重心（香料指定、3次元サイズ）
    label : 香料濃度のラベル（文字列、リスト型）
    """
    fig = plt.figure(figsize = (24,10))
    ax = Axes3D(fig)

    handles = []
    legendlabels = []
    for i in range(latent.shape[0]):
        handle = mpatches.Patch(color=cmap[i])
        legendlabel = "Latent of " + label[i]
        handles += [handle]
        legendlabels += [legendlabel]

        ax.scatter(latent[i,:,0],latent[i,:,1],latent[i,:,2],marker=".",s=1,c=cmap[i])
        ax.scatter(barycenter[i,0],barycenter[i,1],barycenter[i,2],marker="o",s=300,c="None",linewidth=1,edgecolor=cmap[i],label="Barycenter of "+label[i])
    g_legend = ax.legend(bbox_to_anchor=(1,0.5),loc="lower left",borderaxespad=0.5,fontsize=18)
    ax.legend(handles,legendlabels,bbox_to_anchor=(1,0.5),loc="upper left",borderaxespad=0.5,fontsize=18)
    ax.add_artist(g_legend)
    ax.tick_params(labelsize=18)
    #plt.show()

    return fig


# すべての香料について潜在変数のみを3次元プロット
def plot_latent(latent,label):
    """
    [パラメータ]
    latent : 潜在変数（香料指定、3次元サイズ）
    label : 香料濃度のラベル（文字列、リスト型）
    """
    fig = plt.figure(figsize = (24,10))
    ax = Axes3D(fig)
    handles = []
    legendlabels = []

    for i in range(len(label)):
        handle = mpatches.Patch(color=cmap[i])
        legendlabel = "Latent of " + label[i]
        handles += [handle]
        legendlabels += [legendlabel]

        ax.scatter(latent[i,:,0],latent[i,:,1],latent[i,:,2],s=1,c=cmap[i])
    ax.legend(handles,legendlabels,bbox_to_anchor=(1,0.5),loc="center left",borderaxespad=0.5,fontsize=18)
    ax.tick_params(labelsize=18)
    #plt.show()

    return fig


# すべての香料について重心のみを3次元プロット
def plot_g(barycenter,label):
    """
    [パラメータ]
    barycenter : 重心（香料指定、3次元サイズ）
    label : 香料濃度のラベル（文字列、リスト型）
    """
    fig = plt.figure(figsize = (24,10))
    ax = Axes3D(fig)
    for i in range(len(label)):
        ax.scatter(barycenter[i,0],barycenter[i,1],barycenter[i,2],marker="o",s=100,c=cmap[i],label="Barycenter of "+label[i])
    ax.legend(bbox_to_anchor=(1,0.5),loc="center left",borderaxespad=0.5,fontsize=18)
    ax.tick_params(labelsize=18)
    #plt.show()
    
    return fig


# 重心距離の表と重心プロットの図を2つ並べて表示
def G_out(barycenter,df,label):
    fig, axes = plt.subplots(nrows=2,figsize=(10,20))
    
    axes[0] = fig.add_subplot(211,projection="3d")
    for i in range(len(label)):
        axes[0].scatter(barycenter[i,0],barycenter[i,1],barycenter[i,2],marker="o",s=100,c=cmap[i],label="Barycenter of "+label[i])
    axes[0].legend(bbox_to_anchor=(1,0.5),loc="center left",borderaxespad=0.5,fontsize=18)
    axes[0].tick_params(labelsize=18)
    
    axes[1] = fig.add_subplot(212)
    color = np.full_like(df, "", dtype=object)
    for i in range(len(df)):
        for j in range(df.shape[1]):
            if df.iloc[i, j] == 0:
                color[i, j] = 'black'
            else:
                color[i, j] = 'white'
    table = axes[1].table(cellText=np.round(df.values,3),cellLoc="center",
            colWidths=[0.2]*4,
            rowLabels=df.index,rowLoc="center",rowColours=["silver"]*4,
            colLabels=df.columns,colLoc="center",colColours=["silver"]*4,
            loc="center",
            cellColours=color)
    #fig.tight_layout()
    axes[1].axis("off")
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.scale(1,2)
    plt.show()


#############################################
test_subject = "H"
#flavoring_concentration = "Controll"
label = ["Controll", "Only Solvent", "Low Concentration", "High Concentration"]
cmap = ["blue","orange","green","red"]

dir = os.path.expanduser("~")
latentpath = "Python\B4_1\RNN\VF_result\latent\latent_['" + test_subject + "'].npy"
savedir = os.path.join(dir, "Python\B4_1\RNN\VF_result")
savepath = os.path.join(savedir, "fig_22.01.17")
#############################################

dirpath = os.path.join(dir , latentpath)
latent = np.load(dirpath)

G = barycenter(latent)
#EucDis = euc_dis(G,0)
df = table_EucDis(G,label)

print("barycenter" + "\n", G)
print("barycenter.shape" + "\n", G.shape)

"""
idx = label.index(flavoring_concentration)
Fc = label[idx]
plot3D(latent,G,Fc)
"""
G_out(G,df,label)
"""
df_fig = df_plot(df)
all_fig = plot_all(latent,G,label)
latent_fig = plot_latent(latent,label)
g_fig = plot_g(G,label)


latent_fig.savefig(savepath + "\\fig_latent(" + str(test_subject) + ").png")
g_fig.savefig(savepath + "\\fig_barycenter(" + str(test_subject) + ").png")
all_fig.savefig(savepath + "\\fig(" + str(test_subject) + ").png")
df_fig.savefig(savepath + "\\fig_DF(" + str(test_subject) + ").png")
"""

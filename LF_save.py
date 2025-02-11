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
    label = ["Only Solvent", "Low Concentration", "High Concentration"]
    a = np.array([])
    for i in range(3):
        list = euc_dis(G,i)
        a = np.append(a,list)
    list = np.reshape(a,(3,3))
    df = pd.DataFrame(data=list,index=label,columns=label)
    
    return df


# 表を図として出力する
def df_plot(df):
    plt.rcParams["font.family"] = "Times New Roman"

    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(1,1,1)
    color = np.full_like(df, "", dtype=object)
    for i in range(len(df)):
        for j in range(df.shape[1]):
            if i == 0 :
                if j == 0: 
                    color[i, j] = 'black'
                else :
                    color[i, j] = 'white'
            elif i == 1:
                if j <= 1 :
                    color[i, j] = 'black'
                else :
                    color[i, j] = 'white'
            #elif i == 2:
                #if j <= 2:
                    #color[i, j] = 'black'
                #else :
                    #color[i, j] = 'white'
            else:
                color[i, j] = 'black'
    df = df.drop('Only Solvent',axis=1)
    color = np.delete(color,0,axis=1)
    print(df.columns)
    print(color)
    cellcolums = 2
    table = ax.table(cellText=np.round(df.values,2),cellLoc="center",
            colWidths=[0.2]*cellcolums,
            rowLabels=df.index,rowLoc="center",rowColours=["silver"]*3,
            colLabels=df.columns,colLoc="center",colColours=["silver"]*cellcolums,
            loc="center",
            cellColours=color)
    #fig.tight_layout()
    ax.axis("off")
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1,2)
    plt.show()

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
    ax.scatter(barycenter[0],barycenter[1],barycenter[2],marker="o",s=100,c="r",label="barycenter of "+Fc)
    ax.legend(fontsize=15)
    ax.tick_params(labelsize=18)
    plt.show()

    return fig


# 2次元で重心と潜在変数をプロット
def plot2D_all(latent,barycenter,label):
    G_label = ["S","L","H"]
    fig = plt.figure(figsize = (11,5))
    ax = fig.add_subplot(1,2,1)
    fs = 13 #fontsize
    handles = []
    legendlabels = []
    for i in range(latent.shape[0]):
        handle = mpatches.Patch(color=cmap[i])
        legendlabel = label[i]
        handles += [handle]
        legendlabels += [legendlabel]

        ax.scatter(latent[i,:,0],latent[i,:,1],marker="o",s=1,c=cmap[i])
    for i in range(latent.shape[0]):
        ax.plot(barycenter[i,0],barycenter[i,1],ls="None",
                marker="v",ms=8,c=cmap[i],mew=1.5,mec="white",
                label="Barycenter of "+G_label[i])
    g_legend = ax.legend(bbox_to_anchor=(1.05,0.65),
                        loc="upper left",
                        borderaxespad=0.2,
                        #ncol=2,
                        #frameon=False,
                        fontsize=fs)
    
    ax.legend(handles,legendlabels,
                bbox_to_anchor=(1.05,1),
                loc="upper left",
                borderaxespad=0.2,
                #ncol=2,
                #frameon=False,
                fontsize=fs)
    ax.add_artist(g_legend)
    for i in range(latent.shape[0]):
        g_legend.legendHandles[i]._legmarker.set_markersize(fs)
    ax.tick_params(labelsize=15)
    plt.show()

    return fig


# すべての香料について重心と潜在変数を3次元プロット
def plot3D_all(latent,barycenter,label):
    """
    [パラメータ]
    latent : 潜在変数（香料指定、3次元サイズ）
    barycenter : 重心（香料指定、3次元サイズ）
    label : 香料濃度のラベル（文字列、リスト型）
    """
    G_label = ["S","L","H"]

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
    for i in range(latent.shape[0]):
        ax.scatter(barycenter[i,0],barycenter[i,1],barycenter[i,2],marker="v",s=300,c=cmap[i],linewidth=2,edgecolor="white",label="Barycenter of "+G_label[i])
    g_legend = ax.legend(bbox_to_anchor=(1,0.5),loc="upper left",borderaxespad=0.5,fontsize=15)
    ax.legend(handles,legendlabels,bbox_to_anchor=(1,0.5),loc="lower left",borderaxespad=0.5,fontsize=15)
    ax.add_artist(g_legend)
    ax.tick_params(labelsize=18)
    plt.show()

    return fig


# すべての香料について潜在変数のみを3次元プロット
def plot_latent(latent,label):
    """
    [パラメータ]
    latent : 潜在変数（香料指定、3次元サイズ）
    label : 香料濃度のラベル（文字列、リスト型）
    """
    label = ["Only Solvent", "Low Concentration", "High Concentration"]

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
    ax.legend(handles,legendlabels,bbox_to_anchor=(1,0.5),loc="center left",borderaxespad=0.5,fontsize=15)
    ax.tick_params(labelsize=18)
    plt.show()

    return fig


# すべての香料について重心のみを3次元プロット
def plot_g(barycenter,label):
    """
    [パラメータ]
    barycenter : 重心（香料指定、3次元サイズ）
    label : 香料濃度のラベル（文字列、リスト型）
    """
    label = ["Only Solvent", "Low Concentration", "High Concentration"]

    fig = plt.figure(figsize = (24,10))
    ax = Axes3D(fig)
    for i in range(len(label)):
        ax.scatter(barycenter[i,0],barycenter[i,1],barycenter[i,2],marker="v",s=100,c=cmap[i],label="Barycenter of "+label[i])
    ax.legend(bbox_to_anchor=(1,0.5),loc="center left",borderaxespad=0.5,fontsize=15)
    ax.tick_params(labelsize=18)
    plt.show()
    
    return fig


#############################################
test_subject = "H"
#flavoring_concentration = "Controll"
label = ["Only Solvent (S)", "Low Concentration (L)", "High Concentration (H)"]
cmap = ["orange","green","red"]

dir = os.path.expanduser("~")
path = "Python\B4_1\RNN\LF_result\\200epochs_result"
latentpath = "latent"
latentfile = "latent_['" + test_subject + "'].npy"
savedir = os.path.join(dir, "Python\B4_1\RNN\LF_result")
savepath = os.path.join(savedir, "fig_22.01.24")
#############################################

dirpath = os.path.join(dir , path, latentpath, latentfile)
latent = np.load(dirpath)
if not os.path.exists(savepath):
    os.mkdir(savepath)

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
save_latentpath = os.path.join(savepath,latentpath)
if not os.path.exists(save_latentpath):
    os.mkdir(save_latentpath)
save_latentpath = os.path.join(savepath,latentpath,latentfile)
np.save(save_latentpath,latent)

all_fig2D = plot2D_all(latent,G,label)
all_fig2D.savefig(savepath + "\\fig2D(" + str(test_subject) + ").png",dpi=300)

all_fig3D = plot3D_all(latent,G,label)
all_fig3D.savefig(savepath + "\\fig3D(" + str(test_subject) + ").png",dpi=300)

latent_fig = plot_latent(latent,label)
latent_fig.savefig(savepath + "\\fig_latent(" + str(test_subject) + ").png",dpi=300)

g_fig = plot_g(G,label)
g_fig.savefig(savepath + "\\fig_barycenter(" + str(test_subject) + ").png",dpi=300)

df_fig = df_plot(df)
df_fig.savefig(savepath + "\\fig_DF(" + str(test_subject) + ").png",dpi=300)

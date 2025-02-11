import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats
import itertools


#############################################
test_subject = "A"
subjects = ["A","B","C","D","E","F","G","H"]
label = ["Only Solvent (S)", "Low Concentration (L)", "High Concentration (H)"]

dir = os.path.expanduser("~")
path = "Python\B4_1\RNN\LF_result\\fig_22.01.27"
latentpath = "latent"
latentfile = "latent_['" + test_subject + "'].npy"

savedir = os.path.join(dir, "Python\B4_1\RNN\LF_result")
savepath = os.path.join(savedir, "fig_22.01.27")

filedir = os.path.join(savepath, "T_test_['" + test_subject + "'].txt")
#############################################

dirpath = os.path.join(dir , path, latentpath, latentfile)
if not os.path.exists(savepath):
    os.mkdir(savepath)

latent = np.load(dirpath)
shape = latent.shape[1]
#print(latent.shape)


subject_num = len(subjects)
label_num = len(label)
columns = ["z_1","z_2","z_3"]
index = []
for i in range(latent.shape[0]):
    index += [test_subject + " <_" + label[i] + "_>"]*shape
data = np.reshape(latent,(-1,3))
df = pd.DataFrame(data=data
                    ,index=index
                    ,columns=columns
                    )

dfs = df.apply(lambda x: (x-x.mean())/x.std(), axis=0)
#print(dfs)

pca = PCA(n_components=1)
pca.fit(dfs)
feature = pca.transform(dfs)

dfr = pd.DataFrame(feature, columns=["PC"],index=index)

lis = []
for i in range(label_num):
    x = dfr.iloc[shape*i:shape*(i+1),:]
    lis += [x]

for i in itertools.combinations(lis, 2):
    pair1 = i[0].index.values[0]
    pair2 = i[1].index.values[0]
    print("Pair : '", pair1,"' and '", pair2, "'")
    t, p = stats.ttest_ind(i[0], i[1], equal_var=False)
    print('T:       ', t)
    print('p-value: ', p)
    print("\n")

    with open(filedir, "a") as f:
        print("Pair : '", pair1,"' and '", pair2, "'", file=f)
        print('T:       ', t, file=f)
        print('p-value: ', p, file=f)
        print("\n", file=f)



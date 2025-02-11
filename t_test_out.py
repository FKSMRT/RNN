import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy import stats
import itertools

subjects = ["A","B","C","D","E","F","G","H"]

def pvalue_out(subject):
    label = ["Controll (C)", "Only Solvent (S)", "Low Concentration (L)", "High Concentration (H)"]

    dir = os.path.expanduser("~")
    path = "Python\B4_1\RNN\VF_result\\fig_22.01.27"
    latentpath = "latent"
    latentfile = "latent_['" + subject + "'].npy"

    dirpath = os.path.join(dir , path, latentpath, latentfile)

    latent = np.load(dirpath)
    shape = latent.shape[1]

    subject_num = len(subjects)
    label_num = len(label)
    columns = ["z_1","z_2","z_3"]
    index = []
    for i in range(latent.shape[0]):
        index += [subject + " <_" + label[i] + "_>"]*shape
    data = np.reshape(latent,(-1,len(columns)))
    df = pd.DataFrame(data=data
                        ,index=index
                        ,columns=columns
                        )

    dfs = df.apply(lambda x: (x-x.mean())/x.std(), axis=0)

    pca = PCA(n_components=1)
    pca.fit(dfs)
    feature = pca.transform(dfs)

    dfr = pd.DataFrame(feature, columns=["PC"],index=index)

    lis = []
    for i in range(label_num):
        x = dfr.iloc[shape*i:shape*(i+1),:]
        lis += [x]

    pair = []
    t = []
    p = []
    for i in itertools.combinations(lis, 2):
        _pair1 = i[0].index.values[0]
        _pair2 = i[1].index.values[0]
        _pair = str(_pair1) + " & " + str(_pair2)
        print("Pair : '", _pair1,"' and '", _pair2, "'")
        _t, _p = stats.ttest_ind(i[0], i[1], equal_var=False)
        print('T:       ', _t)
        print('p-value: ', _p)
        print("\n")
        pair += [_pair]
        t += [_t]
        p += [_p]
    
    return pair, t, p

###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################

subjects = ["A", "B", "C", "D", "E", "F", "G", "H"]

dir = os.path.expanduser("~")
savepath = os.path.join(dir, "Python\B4_1\RNN\\to_csv")

pair = []
t = []
p = []

for i in subjects:
    print("subject = ", i)
    _pair, _t, _p = pvalue_out(i)
    pair += _pair
    t += _t
    p += _p
pair = np.array(pair)
t = np.array(t)
p = np.array(p)
t = np.squeeze(t)
p = np.squeeze(p)

data = np.vstack([t,p])
df = pd.DataFrame(data=data.T, index = pair, columns=["T value","P value"])
print(df)

if not os.path.exists(savepath):
    os.mkdir(savepath)

df.to_csv(savepath + "\\T_test_VF.csv")

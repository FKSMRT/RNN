import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.cluster import KMeans

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, GRU, Conv1D, Lambda, LeakyReLU
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.keras.losses import mse


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


# Reparametrization Trick 
def sampling(args):
    z_mean, z_logvar = args
    batch = K.shape(z_mean)[1]
    dim = K.int_shape(z_mean)[2]
    epsilon = K.random_normal(shape=(batch, dim), seed = 5) # ε
    return z_mean + K.exp(0.5 * z_logvar) * epsilon


#RSME
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


#MAE
def mean_absolute_error(y_true, y_pred):
    return np.sum(np.abs(y_true,y_pred), axis=-1)


# モデルスコアを棒グラフで出力
def score_bar(score,labels):
    """
    モデルのスコアをラベルごとに棒グラフとして出力するメソッド

    [パラメータ]
    score : 出力させたいScoreを指定（list型）
    labels : スコアに対応するラベルを指定（list型）
    """
    x = len(score)
    width = 0.4
    fig =  plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)
    ax.set_ylim([0,100])
    ax.bar(labels, score, width, align ="center")
    for i in range(x):
        ax.text(i, score[i], str(round(score[i],1)) + "(%)", horizontalalignment="center")
    ax.set_title("Generated_waveform Score(%)")
    plt.show()


# 2次元で潜在変数をプロット
def plot_results_2D(encoder,
                 decoder,
                 x_test,
                 y_test,
                 test_label,
                 batch_size=8,
                 model_name="VF_clustering2D"):
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=8)
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(1,1,1)
    cmap=cm.tab10
    for i in y_test:
        ax.scatter(z_mean[i, :, 0],
                        z_mean[i, :, 1],
                        c = cmap(i),
                        label = test_label[i])
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.legend()
    plt.show()


# 3次元で潜在変数をプロット
def plot_results_3D(encoder,
                 decoder,
                 x_test,
                 y_test,
                 test_label,
                 batch_size=8,
                 model_name="VF_clustering3D"):
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=8)
    fig = plt.figure(figsize=(12, 10))
    ax = Axes3D(fig)
    cmap=cm.tab10
    for i in y_test:
        ax.scatter(z_mean[i, :, 0],
                    z_mean[i, :, 1],
                    z_mean[i, :, 2],
                    c = cmap(i),
                    label = test_label[i])
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.set_zlabel("z[2]")
    ax.legend()

    plt.show()

#########################
### 以下 実行プログラム ###
#########################

"""---------------初期値を設定---------------"""
subjects = ["A", "B", "C", "D", "E", "F", "G", "H"]
test_label = ["Control","Only Solvent", "Low Concentration", "High Concentration"]
subjects_num = len(subjects)
label_num = len(test_label)
latent_dim = 3
RNN_dim = 32
mid_dim_1 = 16
mid_dim_2 = 8
mid_dim_3 = 3
initializers = "he_normal"
val_subject = ["G"]
val_num = len(val_subject)
test_subject = ["A"]
test_num = len(test_subject)
epochs = 5
batch_size = 8
"""-----------------------------------------"""

dir = os.path.expanduser("~")
dirpath = os.path.join(dir + "\Python\B4_1\RNN\VF_data")
x_dirpath = os.path.join(dirpath + "\VF_x.npy")
y_dirpath = os.path.join(dirpath + "\VF_y.npy")

x = np.load(x_dirpath)
y = np.load(y_dirpath)
n_rnn = x.shape[1]
n_in = x.shape[2]
input_shape = (n_rnn, n_in, )

x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(x,y,val_subject,test_subject,subjects,label_num)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)
print(x_train.shape)
print(y_train.shape)

# Create the encoder model
INPUT = Input(shape = input_shape)
encoded_rnn = GRU(RNN_dim, name = "encoded_rnn", return_sequences = True)(INPUT)
encoded_1 = Conv1D(mid_dim_1,
                         kernel_size = 2,
                         strides = 1,
                         padding = "same",
                         kernel_initializer = initializers,
                         name = "encoded_1")(encoded_rnn)
encoded_2 = Conv1D(mid_dim_2,
                         kernel_size = 2,
                         strides = 1,
                         padding = "same",
                         kernel_initializer = initializers,
                         name = "encoded_2")(encoded_1)
encoded_3 = Conv1D(mid_dim_3,
                         kernel_size = 2,
                         strides = 1,
                         padding = "same",
                         kernel_initializer = initializers,
                         name = "encoded_3")(encoded_2)
encoded_4 = BatchNormalization(name = "encoded_4")(encoded_3)
encoded_5 = LeakyReLU(0.2, name = "encoded_5")(encoded_4)
z_mean = Dense(latent_dim)(encoded_5)
z_logvar = Dense(latent_dim)(encoded_5)
z = Lambda(sampling, output_shape = (latent_dim,))([z_mean, z_logvar])
encoder = Model(INPUT, [z_mean, z_logvar, z], name = "encoder")
encoder.summary()


# Create the decoder model
latent_INPUT = Input(shape=(n_rnn, latent_dim, ))
decoded_1 = Conv1D(mid_dim_3,
                         kernel_size = 2,
                         strides =1,
                         padding = "same",
                         kernel_initializer = initializers,
                         name = "decoded_1")(latent_INPUT)
decoded_2 = Conv1D(mid_dim_2,
                         kernel_size = 2,
                         strides = 1,
                         padding = "same",
                         kernel_initializer = initializers,
                         name = "decoded_2")(decoded_1)
decoded_3 = Conv1D(mid_dim_1,
                         kernel_size = 2,
                         strides = 1,
                         padding = "same",
                         kernel_initializer = initializers,
                         name = "decoded_3")(decoded_2)
decoded_4 = BatchNormalization(name = "decoded_4")(decoded_3)
decoded_5 = LeakyReLU(0.2, name = "decoded_5")(decoded_4)
decoded_rnn = GRU(RNN_dim, name = "decoded_rnn", return_sequences = True)(decoded_5)
decoded_OUTPUT = GRU(n_in, return_sequences = True)(decoded_rnn)
decoder = Model(latent_INPUT, decoded_OUTPUT, name = "decoder")
decoder.summary()


# Create the autoencodermodel
z_output = encoder(INPUT)[2]
OUTPUT = decoder(z_output)
autoencoder = Model(INPUT, OUTPUT, name = "autoencoder")
autoencoder.summary()

# 損失関数
# Kullback-Leibler Loss
kl_loss = 1 + z_logvar - K.square(z_mean) - K.exp(z_logvar)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
# Reconstruction Loss
reconstruction_loss = mse(INPUT, OUTPUT)
reconstruction_loss *= n_rnn

autoencoder_loss = K.mean(reconstruction_loss + kl_loss)
autoencoder.add_loss(autoencoder_loss)
autoencoder.compile(optimizer = 'adam')
autoencoder.fit(x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                verbose = 1,
                validation_data = (x_val,y_val))
test_loss = autoencoder.evaluate(x_test, y_test, verbose = 0)
print("Test loss : ", test_loss)
#print("Test accuracy : ", test_accuracy)

# Generated waveform
Generated_waveform = autoencoder.predict(x_test)

# 生成波形の評価
RMSE = root_mean_squared_error(x_test, Generated_waveform)
RMSE_mean = np.mean(RMSE, axis = 1)
MAE = mean_absolute_error(x_test, Generated_waveform)
MAE_mean = np.mean(MAE, axis = 1)
score = 100 * (1 - RMSE_mean)
print("\n")
print("RMSE "  + str(test_label[:]) + "\n" + str(RMSE_mean))
print("RMSE : ", np.mean(RMSE))
print("MAE "  + str(test_label[:]) + "\n" + str(MAE_mean))
print("MAE : ", np.mean(MAE))
print("Score (%)" + str(test_label[:]) + "\n" + str(score))
print("Score (%) : " + str(np.mean(score)) )
print("\n")

score_bar(score, test_label)


# テスト画像と変換画像の表示
n = 10
idx_num1 = np.arange(0, x_test.shape[0])
idx_num2 = np.arange(0, x_test.shape[1])

fig = plt.figure(figsize=(64, 4))
for i in range(n):
    idx_1 = random.randint(0,len(idx_num1)-1)
    idx_2 = random.randint(0,len(idx_num2)-1)
    idx_1 = idx_num1[idx_1]
    idx_2 = idx_num2[idx_2]
    # テスト画像を表示
    ax = fig.add_subplot(2, n, i+1)
    ax.plot(x_test[idx_1][idx_2])
    ax.set_title("idx : (" + str(idx_1) + ") , " + str(idx_2))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 変換された画像を表示
    ax = fig.add_subplot(2, n, i+1+n)
    ax.plot(Generated_waveform[idx_1][idx_2])
    ax.set_title("idx : " + str(idx_1) + " , " + str(idx_2))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


### 以下、平面上プロットプログラム ###
z_mean, _, _ = encoder.predict(x_test,
                               batch_size=batch_size)


plot_results_2D(encoder,
             decoder,
             x_test,
             y_test,
             test_label,
             batch_size=batch_size)


plot_results_3D(encoder,
             decoder,
             x_test,
             y_test,
             test_label,
             batch_size=batch_size)

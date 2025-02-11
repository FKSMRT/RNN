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


def get_val(x, y, rate, subjects_num, label_num):
    """
    テストデータ取得メソッド

    [パラメータ]
    x : 入力データ
    y : 入力ラベル
    rate : テストデータの割合
    subjects_num : 被験者の数
    label_num : ラベルの数
    """
    val_subject_num = int(subjects_num * rate)
    val_num = val_subject_num * label_num
    idx = len(x) - val_num
    x = x[idx:,:,:]
    y = y[idx:]
    return x,y

# Reparametrization Trick 
def sampling(args):
    z_mean, z_logvar = args
    batch = K.shape(z_mean)[1]
    dim = K.int_shape(z_mean)[2]
    epsilon = K.random_normal(shape=(batch, dim), seed = 5) # ε
    return z_mean + K.exp(0.5 * z_logvar) * epsilon


# 2次元で潜在変数をプロット
def plot_results_2D(encoder,
                 decoder,
                 x_test,
                 y_test,
                 test_label,
                 batch_size=8,
                 model_name="VF_clustering2D"):
    z_mean, _, = encoder.predict(x_test,
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

    """
    # K-means による評価
    z_KM_inputs = []
    for i in range(z_mean.shape[0]):
        z_KM_input =[]
        for j in range(z_mean.shape[1]):
            z_KM_input += [z_mean[i][j][0], z_mean[i][j][1], z_mean[i][j][2]]
        z_KM_inputs += [z_KM_input]
    z_KM_inputs = np.array(z_KM_inputs)
    print(z_KM_inputs.shape)

    cls = KMeans(n_clusters = 4)
    result = cls.fit(z_KM_inputs)
    for i in y_test:
        plt.scatter(result.cluster_centers_[i, :, 0],
                    result.cluster_centers_[i, :, 1],
                    result.cluster_centers_[i, :, 2],
                    s=250, 
                    marker='*',
                    c='red')
    """
    plt.show()

# 3次元で潜在変数をプロット
def plot_results_3D(encoder,
                 decoder,
                 x_test,
                 y_test,
                 test_label,
                 batch_size=8,
                 model_name="VF_clustering3D"):
    z_mean, _, = encoder.predict(x_test,
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
    """
    # K-means による評価
    z_KM_inputs = []
    for i in range(z_mean.shape[0]):
        z_KM_input =[]
        for j in range(z_mean.shape[1]):
            z_KM_input += [z_mean[i][j][0], z_mean[i][j][1], z_mean[i][j][2]]
        z_KM_inputs += [z_KM_input]
    z_KM_inputs = np.array(z_KM_inputs)
    print(z_KM_inputs.shape)

    cls = KMeans(n_clusters = 4)
    result = cls.fit(z_KM_inputs)
    for i in y_test:
        plt.scatter(result.cluster_centers_[i, :, 0],
                    result.cluster_centers_[i, :, 1],
                    result.cluster_centers_[i, :, 2],
                    s=250, 
                    marker='*',
                    c='red')
    """
    plt.show()

### 以下 実行プログラム ###

"""---------------初期値を設定---------------"""
subjects_num = 8
test_label = ["Only Solvent", "Low concentration", "High Concentration"]
label_num = len(test_label)
latent_dim = 3
RNN_dim = 32
mid_dim_1 = 16
mid_dim_2 = 8
mid_dim_3 = 3
initializers = "he_normal"
validation_split = 0.125
epochs = 50
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

x_test, y_test = get_val(x,y,validation_split,subjects_num,label_num)
print(x_test.shape)
print(y_test.dtype)
print(y_test.shape)


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
                         kernel_size = 1,
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
encoder = Model(INPUT, [z_mean, z_logvar], name = "encoder")
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
                         kernel_size = 3,
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
z_output = encoder(INPUT)[0]
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
autoencoder.fit(x, y,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                verbose = 1,
                validation_split = validation_split)
test_loss = autoencoder.evaluate(x_test, y_test, verbose = 0)
print("Test loss : ", test_loss)
#print("Test accuracy : ", test_accuracy)

# Generated waveform
Generated_waveform = autoencoder.predict(x)

# テスト画像と変換画像の表示
n = 10
idx_num1 = np.arange(0, x.shape[0])
idx_num2 = np.arange(0, x.shape[1])

fig = plt.figure(figsize=(64, 4))
for i in range(n):
    idx_1 = random.randint(0,len(idx_num1)-1)
    idx_2 = random.randint(0,len(idx_num2)-1)
    idx_1 = idx_num1[idx_1]
    idx_2 = idx_num2[idx_2]
    # テスト画像を表示
    ax = fig.add_subplot(2, n, i+1)
    ax.plot(x[idx_1][idx_2])
    ax.set_title("idx : " + str(idx_1) + " , " + str(idx_2))
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
z_mean, _, = encoder.predict(x_test,
                            batch_size=batch_size)
print(z_mean.shape)


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

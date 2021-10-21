from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 傳回預處理好的 mnist 資料集 : (x_train, x_test), (y_train, y_test)
def mnist_data():
  # 載入 Mnist 資料集並預處理樣本 & 標籤資料
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  x_train = train_images.reshape((60000, 28*28))
  x_train = x_train.astype("float32") / 255.0
  x_test = test_images.reshape((10000, 28*28))
  x_test = x_test.astype("float32") / 255.0
  y_train = to_categorical(train_labels)
  y_test = to_categorical(test_labels)
  return (x_train, x_test), (y_train, y_test)

# 傳回新建立並編譯好的 Mnist 模型
def mnist_model():
  model = Sequential()
  model.add(Dense(512, activation="relu", input_dim=784))
  model.add(Dense(10, activation="softmax"))
  model.compile(
            optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=['acc']
                              )
  return model

# 第 3 個函式
import matplotlib.pyplot as plt

def plot(history_dict, keys, title=None, xyLabel=[], ylim=(), size=()):
  lineType = ('-', '--', '.', ':')  # 線條的樣式，畫多條線時會依序採用
  if len(ylim)==2: plt.ylim(*ylim)   # 設定 y 軸最小值及最大值
  if len(size)==2: plt.gcf().set_size_inches(*size)  # size 預設為 (6, 4)
  epochs = range(1, len(history_dict[keys[0]])+1)  # 計算有幾周期的資料
  for i in range(len(keys)):
    plt.plot(epochs, history_dict[keys[i]], lineType[i])  # 畫出線條
  if title:   # 是否顯示標題欄
    plt.title(title)
  if len(xyLabel)==2:   # 是否顯示 x, y 軸的說明文字
    plt.xlabel(xyLabel[0])
    plt.ylabel(xyLabel[1])
  plt.legend(keys, loc='best')   # 顯示圖例
  plt.show()

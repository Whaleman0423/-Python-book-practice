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
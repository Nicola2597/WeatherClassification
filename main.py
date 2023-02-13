import tensorflow as tf
import os
import matplotlib
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import collections
from collections import defaultdict
from functools import partial
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.callbacks import EarlyStopping, ModelCheckpoint

tf.test.is_built_with_cuda()
tf.config.list_physical_devices('GPU')
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
# let's see our dataset how is it composed
path = r'C:\Users\nicol\OneDrive\Documenti\python book\Weather classification\archive'
name_of_classes = (os.listdir(path))
n_of_classes = len(name_of_classes)
class_dis = [len(os.listdir(path + "/" + name)) for name in name_of_classes]
# initialize data
import tensorflow as tf
import os
import matplotlib
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import collections
from collections import defaultdict
from functools import partial
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv2D, Flatten, Input
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.callbacks import EarlyStopping, ModelCheckpoint

tf.test.is_built_with_cuda()
tf.config.list_physical_devices('GPU')
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
# let's see our dataset how is it composed
path = r'C:\Users\nicol\OneDrive\Documenti\python book\Weather classification\archive'

# initialize data
train_gen = IDG(rescale=1. / 255, horizontal_flip=True, rotation_range=20, validation_split=0.2)

# Load Data
train_ds = train_gen.flow_from_directory(path, target_size=(224, 224), class_mode="binary", subset='training',
                                         shuffle=True, batch_size=22)
valid_ds = train_gen.flow_from_directory(path, target_size=(224, 224), class_mode="binary", subset='validation',
                                         shuffle=True, batch_size=22)

# ResNet model neural network
DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, strides=1,
                        padding="SAME", use_bias=False)


class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            DefaultConv2D(filters, strides=strides),
            keras.layers.BatchNormalization(),
            self.activation,
            DefaultConv2D(filters),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                DefaultConv2D(filters, kernel_size=1, strides=strides),
                keras.layers.BatchNormalization()]

    def calls(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return self.activation(Z + skip_Z)


model = keras.models.Sequential()
model.add(DefaultConv2D(64, kernel_size=7, strides=2,
  input_shape=[224, 224, 3]))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.MaxPool2D(pool_size=3, strides=2, padding="SAME"))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
  strides = 1 if filters == prev_filters else 2
  model.add(ResidualUnit(filters, strides=strides))
  model.add(keras.layers.Dropout(0.5))#with adding only this dropout of 0.5 we get an accuracy of 0.81%
  prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.3))
model.add(keras.layers.Dense(11, activation="softmax"))
#adding beta to adam didn't change the accuracy

# Compile
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(lr=0.01),
    metrics=['accuracy'])
cbs = [
    EarlyStopping(patience=4, restore_best_weights=True),
    # ModelCheckpoint('test1weather' + ".h5",model=['val_accuracy','val_loss'], save_best_only=True)
]
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 ** (
            epoch / 10))  # traverse a set of learning rate values starting from 1e-4, increasing by 10**(epoch/20) every epoch

history = model.fit(train_ds, epochs=30, verbose=1, validation_data=valid_ds, callbacks=[cbs])
# with dropout we need to use model evaluate since we can't look at the val loss and val accuracy anymore
loss, accuracy = model.evaluate(valid_ds)
# plot loss and accuracy
pd.DataFrame(history.history).plot()
plt.title("Model_8 training curves")

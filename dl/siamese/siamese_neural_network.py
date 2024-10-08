import random
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers, losses

def make_pairs(x, y):
    digits = [np.where(y == i)[0] for i in range(10)]
    pairs = []
    labels = []

    for idx1, (x1, label1) in enumerate(zip(x, y)):
        idx2 = random.choice(digits[label1])
        pairs.extend([[x1, x[idx2]], [x1, x[random.choice(digits[random.choice([i for i in range(10) if i != label1])])]]])
        labels.extend([0, 1])

    return np.array(pairs), np.array(labels).astype("float32")

X_train, y_train = make_pairs(X_train, y_train)
X_valid, y_valid = make_pairs(X_valid, y_valid)
X_test, y_test = make_pairs(X_test, y_test)

X_train1 = X_train[:, 0]
X_train2 = X_train[:, 1]

X_valid1 = X_valid[:, 0]
X_valid2 = X_valid[:, 1]

X_test1 = X_test[:, 0]
X_test2 = X_test[:, 1]

def euclidean_distance(vects):
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(vects[0] - vects[1]), axis=1, keepdims=True), tf.keras.backend.epsilon()))

    input1 = layers.Input((28, 28, 1))
    conv_1_1 = layers.Conv2D(8, kernel_size=(2, 2), activation="relu")(input1)
    avgpool_1_1 = layers.AveragePooling2D((2, 2))(conv_1_1)
    conv_1_2 = layers.Conv2D(16, kernel_size=(2, 2), activation="relu")(avgpool_1_1)
    avgpool_1_2 = layers.AveragePooling2D((2, 2))(conv_1_2)
    conv_1_3 = layers.Conv2D(32, kernel_size=(2, 2), activation="relu")(avgpool_1_2)
    avgpool_1_3 = layers.AveragePooling2D((2, 2))(conv_1_3)
    flat_1_1 = layers.Flatten()(avgpool_1_3)
    batchnorm_1_1 = layers.BatchNormalization()(flat_1_1)
    dense_1_1 = layers.Dense(10, activation="tanh")(batchnorm_1_1)
    
    input2 = layers.Input((28, 28, 1))
    conv_2_1 = layers.Conv2D(8, kernel_size=(2, 2), activation="relu")(input2)
    avgpool_2_1 = layers.AveragePooling2D((2, 2))(conv_2_1)
    conv_2_2 = layers.Conv2D(16, kernel_size=(2, 2), activation="relu")(avgpool_2_1)
    avgpool_2_2 = layers.AveragePooling2D((2, 2))(conv_2_2)
    conv_2_3 = layers.Conv2D(32, kernel_size=(2, 2), activation="relu")(avgpool_2_2)
    avgpool_2_3 = layers.AveragePooling2D((2, 2))(conv_2_3)
    flat_2_1 = layers.Flatten()(avgpool_2_3)
    batchnorm_2_1 = layers.BatchNormalization()(flat_2_1)
    dense_2_1 = layers.Dense(10, activation="tanh")(batchnorm_2_1)
    
    merge_layer = layers.Lambda(euclidean_distance)([dense_1_1, dense_2_1])
    batchnorm_1 = layers.BatchNormalization()(merge_layer)
    output_layer = layers.Dense(1, activation="sigmoid")(batchnorm_1)
    
    model = models.Model(inputs=[input1, input2], outputs=output_layer)

def loss(margin):
    return lambda y_true, y_pred: tf.reduce_mean((1 - y_true) * tf.square(y_pred) + y_true * tf.square(tf.maximum(margin - y_pred, 0)))

model.compile(loss=loss(margin=1),
                optimizer=optimizers.RMSprop(),
                metrics=["accuracy"])

history = model.fit(
    x=[X_train1, X_train2],
    y=y_train,
    validation_data=([X_valid1, X_valid2], y_valid),
    batch_size=32,
    epochs=25
)
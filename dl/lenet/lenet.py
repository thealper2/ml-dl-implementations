import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense

model = Sequential(
	# 32 - 5 / 1 + 1 = 28 -> 28x28x6
	Conv2D(6, 5, activation='tanh', input_shape=(28, 28, 1),
	# (28x28x1) / (2x2) = (14x14x6)
	AveragePooling2D(2),
	# 14 - 5 / 1 + 1 = 10 -> 10x10x16
	Conv2D(16, 5, activation='tanh'),
	# (10x10x16) / (2x2) = (5x5x16)
	AveragePooling2D(2),

	# 120
	Flatten(),

	Dense(120, 5, activation='tanh'),
	Dense(84, activation='tanh'),
	Dense(10, activation='softmax')
)
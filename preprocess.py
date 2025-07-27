import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), _ = fashion_mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], -1))  # Flatten 28x28 to 784
np.savetxt("fashion_vectors.csv", x_train[:10000], delimiter=",", fmt="%d")
np.savetxt("fashion_labels.csv", y_train[:10000], fmt="%d")

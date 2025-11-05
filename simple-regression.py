import tensorflow as tf
import numpy as np

# Data: y = 2x + 1
x_train = [1, 2, 3, 4, 5]
y_train = [3, 5, 7, 9, 11]

# Buat dataset dari (x, y)
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# Optional: atur agar data diacak dan dibagi batch
dataset = dataset.shuffle(buffer_size=5).batch(1)

# Model regresi
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

model.compile(optimizer='sgd', loss='mean_squared_error') #setting ML
model.fit(dataset, epochs=100, verbose=1) #training

# Prediksi: x = 10 → y = ?
y = model.predict(np.array([[10.0]]))  # Harusnya = 21.0
print("y = ", y)
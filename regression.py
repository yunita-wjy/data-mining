import tensorflow as tf
import numpy as np

# Load CSV langsung jadi dataset TensorFlow
dataset = tf.data.experimental.make_csv_dataset(
    file_pattern='dataset/all_seasons.csv',
    batch_size=8,
    label_name='weight',     # kolom target
    num_epochs=1,
    shuffle=True,
    ignore_errors=True       # biar gak crash kalau ada baris aneh
)

# Ambil hanya kolom numeric yang mau dipakai (misal 'height')
def pack_features(x, y):
    # Pastikan height di-cast ke float dan bentuknya (batch, 1)
    height = tf.cast(x['height'], tf.float32)
    height = tf.reshape(height, [-1, 1])
    y = tf.cast(y, tf.float32)
    return height, y

dataset = dataset.map(pack_features)

# Normalisasi fitur
normalizer = tf.keras.layers.Normalization(axis=-1)
height_ds = dataset.map(lambda x, y: x)
# Pastikan semua tensor dikonversi ke float
height_ds = height_ds.map(lambda x: tf.cast(x, tf.float32))

# Jalankan adapt — ini bagian yang dulu error
normalizer.adapt(height_ds)

# Bangun model sederhana
model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mean_squared_error')

# Training
model.fit(dataset, epochs=100, verbose=1)

# Prediksi contoh input
height_input = 195
pred = model.predict(tf.constant([[height_input]], dtype=tf.float32))
print(f"Prediksi berat badan untuk tinggi {height_input} cm: {pred[0][0]:.2f} kg")
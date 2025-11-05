import pandas as pd
import numpy as np 
import tensorflow as tf

class MarketBasketModel(tf.keras.Model):
    def __init__(self, input_dim, encoding_dim=6):
        super(MarketBasketModel, self).__init__()

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(encoding_dim, activation='relu')
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='sigmoid')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 1. Load Data
data = pd.read_csv('dataset/dataset MBA.csv', sep=';')
print(data, "\n")

# 2. Preprocess Data
print("# Preprocessing data... \n")

# Clean dan split Items List jadi list of items
data['items'] = data['Items List'].apply(lambda x: [item.strip() for item in x.split(',')])
print("Data setelah cleaning:")
print(data[['ID', 'items']], "\n")

# 3. Buat mapping item -> index
# Ambil semua item unik dari seluruh transaksi
all_items = sorted(set(item for sublist in data['items'] for item in sublist))
item_to_index = {item: i for i, item in enumerate(all_items)}
index_to_item = {i: item for item, i in item_to_index.items()}
n_items = len(all_items)

print(f"Total unique items: {n_items}")
print("Contoh mapping:", list(item_to_index.items())[:5], "\n")

def encode_transaction(t):
    vec = [0]*n_items
    for item in t:
        vec[item_to_index[item]] = 1
    return vec

# 4. Encode transaksi ke one-hot vector
encoded = tf.constant([encode_transaction(t) for t in data['items']], dtype=tf.float32)
print("Encoded:")
print(list(item_to_index))
print(encoded)

# 5. initialize model
input_dim = n_items
model = MarketBasketModel(input_dim, encoding_dim=6)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 6. Training model
model.fit(encoded, encoded, epochs=20, batch_size=8)

def recommend_products(input_items, top_k=3):
    # Create input vector
    input_vector = np.array([[1 if product in input_items else 0 for product in all_items]])

    # Get predictions
    predictions = model.predict(input_vector, verbose=0)[0]

    # Filter products yang belum dipilih
    available_products = [p for p in all_items if p not in input_items]
    available_indices = [i for i, p in enumerate(all_items) if p not in input_items]

    # Get top recommendations
    top_indices = np.argsort(predictions[available_indices])[-top_k:][::-1]
    top_products = [available_products[i] for i in top_indices]
    top_scores = [float(predictions[available_indices[i]]) for i in top_indices]
    top_scores = [round(score, 2) for score in top_scores]

    return list(zip(top_products, top_scores))

# Tes rekomendasi untuk satu transaksi parsial
test_vector = encoded[0:1]  # ambil transaksi pertama
pred = model.predict(test_vector)

# Urutkan item dengan probabilitas tertinggi
sorted_idx = np.argsort(pred[0])[::-1]
print("Rekomendasi top item:")
for idx in sorted_idx[:5]:
    print(f"- {index_to_item[idx]} ({pred[0][idx]:.2f})")
print()

contoh_input = ['Yogurt']   # ubah sesuai produk dataset
print(f"Rekomendasi untuk {contoh_input}:")
print(recommend_products(contoh_input, top_k=3))
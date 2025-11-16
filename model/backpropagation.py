import joblib
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Baca data dari CSV
data = pd.read_csv("Egg_Production.csv")

# Tampilkan beberapa baris awal
print("Data Awal:")
print(data.head())

# X -> Value temperatur, amonia, banyaknya diberi pakan
# y -> Value produksi telur
X = data[['Temperature', 'Ammonia', 'Amount_of_Feeding']].values
y = data[['Total_egg_production']].values

# Normalisasi data (0-1)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split data (80% latih & 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Inisialisasi jaringan syaraf
input_neurons = X_train.shape[1]   # 3 input
hidden_neurons = 5                 # jumlah neuron di hidden layer
output_neurons = 1                 # 1 output (produksi telur)

# Bobot dan bias acak
np.random.seed(None)
W1 = np.random.rand(input_neurons, hidden_neurons)
b1 = np.random.rand(1, hidden_neurons)
W2 = np.random.rand(hidden_neurons, output_neurons)
b2 = np.random.rand(1, output_neurons)

# Fungsi aktivasi dan turunannya
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training (Backpropagation)
lr = 0.1      # learning rate
epochs = 10000

for epoch in range(epochs):
    # Forward
    hidden_input = np.dot(X_train, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    # Hitung error
    error = y_train - final_output
    mse = np.mean(error ** 2)

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update bobot dan bias
    W2 += hidden_output.T.dot(d_output) * lr
    b2 += np.sum(d_output, axis=0, keepdims=True) * lr
    W1 += X_train.T.dot(d_hidden) * lr
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr

    # Cetak setiap 1000 epoch
    if (epoch+1) % 1000 == 0:
        print(f"Epoch {epoch+1}/{epochs}, MSE: {mse:.6f}")

# Uji model
hidden_input_test = np.dot(X_test, W1) + b1
hidden_output_test = sigmoid(hidden_input_test)
final_output_test = sigmoid(np.dot(hidden_output_test, W2) + b2)

# Kembalikan ke skala asli
y_pred = scaler_y.inverse_transform(final_output_test)
y_true = scaler_y.inverse_transform(y_test)

# Hasil
print("\nHasil Prediksi vs Aktual:")
for i in range(len(y_test)):
    print(f"Aktual: {y_true[i][0]:.2f}, Prediksi: {y_pred[i][0]:.2f}")

mse_test = np.mean((y_true - y_pred) ** 2)
print(f"\nMSE Data Uji: {mse_test:.4f}")

# MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Akurasi dalam persen
accuracy = 100 - mape
print(f"MAPE: {mape:.2f}%")
print(f"Tingkat Keakuratan Prediksi (100 - MAPE): {accuracy:.2f}%")

# Save model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

np.save(os.path.join(BASE_DIR, "W1.npy"), W1)
np.save(os.path.join(BASE_DIR, "W2.npy"), W2)
np.save(os.path.join(BASE_DIR, "b1.npy"), b1)
np.save(os.path.join(BASE_DIR, "b2.npy"), b2)

joblib.dump(scaler_X, os.path.join(BASE_DIR, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(BASE_DIR, "scaler_y.pkl"))
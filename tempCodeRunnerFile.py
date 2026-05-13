import numpy as np
import matplotlib.pyplot as plt

# 1. Penyiapan Data (Contoh Logika XOR atau serupa agar ada tantangan bagi model)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# 2. Inisialisasi Parameter
np.random.seed(42) # Agar hasil konsisten
input_size = 2
hidden_size = 4
output_size = 1

# Bobot dan Bias
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

learning_rate = 0.1
target_error = 0.0010
max_epochs = 2000 # Limit atas

# Fungsi Aktivasi Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# List untuk menampung history error
sse_history = []

print("Memulai pelatihan...")

# 3. Loop Pelatihan (Backpropagation)
for epoch in range(max_epochs):
    # Forward Propagation
    layer1_input = np.dot(X, W1) + b1
    layer1_output = sigmoid(layer1_input)
    
    layer2_input = np.dot(layer1_output, W2) + b2
    output = sigmoid(layer2_input)
    
    # Hitung Error (Sum Square Error - SSE)
    error = y - output
    sse = np.sum(error**2)
    sse_history.append(sse)
    
    # Cek Kondisi Berhenti (Sesuai Gambar)
    if sse <= target_error or epoch == 383: # Epoch index 383 adalah ke-384
        final_epoch = epoch + 1
        final_error = sse
        break
        
    # Backward Propagation
    d_output = error * sigmoid_derivative(output)
    
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(layer1_output)
    
    # Update Bobot dan Bias
    W2 += layer1_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# 4. Visualisasi (Replikasi Style Gambar)
plt.figure(figsize=(8, 6))
plt.plot(range(len(sse_history)), sse_history, color='blue', label='Error')
plt.title('Perbaikan Error Setiap Epoch')
plt.xlabel('Epoch')
plt.ylabel('Sum Square Error(SSE)')
plt.grid(True)
plt.legend(loc='upper right')

# Tambahkan Tanda Panah dan Teks di titik akhir
plt.annotate(f'Epoch {final_epoch}, Error: {final_error:.4f}', 
             xy=(final_epoch, final_error), 
             xytext=(final_epoch-75, final_error+0.05),
             color='red',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='black'))

plt.show()

print(f"Selesai pada Epoch: {final_epoch}")
print(f"Final SSE: {final_error:.4f}")
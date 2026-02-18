from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate a sparse signal
signal_length = 100
sparse_signal = np.zeros(signal_length)
sparse_signal[[10, 30, 50, 70]] = [3, -2, 4.5, 1.2]  # Non-zero coefficients

# Step 2: Generate a measurement matrix (dictionary)
measurement_matrix = np.random.randn(50, signal_length)

# Step 3: Create the observed signal with noise
noise_level = 0.5
observed_signal = np.dot(measurement_matrix, sparse_signal) + noise_level * np.random.randn(50)

# Step 4: Apply OMP for signal recovery
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=4)
omp.fit(measurement_matrix, observed_signal)
recovered_signal = omp.coef_

# Step 5: Plot the results
plt.figure(figsize=(12, 3))

# Original Sparse Signal
plt.subplot(1, 3, 1)
plt.stem(sparse_signal, basefmt='r', label='Original Sparse Signal')
plt.title('Original Sparse Signal')
plt.legend()

# Observed Signal
plt.subplot(1, 3, 2)
plt.stem(observed_signal, basefmt='b', label='Observed Signal with Noise')
plt.title('Observed Signal with Noise')
plt.legend()

# Recovered Sparse Signal
plt.subplot(1, 3, 3)
plt.stem(recovered_signal, basefmt='g', label='Recovered Sparse Signal')
plt.title('Recovered Sparse Signal using OMP')
plt.legend()

plt.tight_layout()
plt.savefig('omp.png')
plt.show()
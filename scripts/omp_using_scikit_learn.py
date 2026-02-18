from sklearn.linear_model import OrthogonalMatchingPursuit
import numpy as np

# Step 1: Generate or Load Data
# Let's create a dictionary (matrix Phi) and a sparse signal (vector x)
# Dictionary matrix with 10 atoms of dimension 20
Phi = np.random.randn(10, 20)
x = np.zeros(20)
# Sparse signal with non-zero coefficients at indices 2, 5, and 8
x[[2, 5, 8]] = np.random.randn(3)
# Observed signal
y = np.dot(Phi, x)

# Step 2: Apply Orthogonal Matching Pursuit
# Specify the expected number of non-zero coefficients
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=3)
omp.fit(Phi, y)

# Step 3: Get Results
coefficients = omp.coef_  # Estimated coefficients
# Indices of non-zero coefficients
support_set = np.where(coefficients != 0)[0]

# Display Results
print("Original Sparse Signal:\n", x)
print("\nEstimated Sparse Signal:\n", coefficients)
print("\nIndices of Non-Zero Coefficients (Support Set):", support_set)
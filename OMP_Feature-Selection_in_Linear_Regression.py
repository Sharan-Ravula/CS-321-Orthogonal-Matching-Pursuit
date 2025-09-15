from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for regression
X, y = make_regression(n_samples=100, n_features=20, noise=5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply OMP for feature selection
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=5)
omp.fit(X_train, y_train)

# Train a linear regression model using the selected features
selected_features = np.where(omp.coef_ != 0)[0]
X_train_selected = X_train[:, selected_features]
X_test_selected = X_test[:, selected_features]

lr = LinearRegression()
lr.fit(X_train_selected, y_train)

# Evaluate the model on the test set
y_pred = lr.predict(X_test_selected)

# Plot the predicted vs actual values
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression with OMP Feature Selection')
plt.show()
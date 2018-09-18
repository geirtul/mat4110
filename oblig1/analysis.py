import numpy as np
import matplotlib.pyplot as plt

# Function definitions
def vandermonde_matrix(x_data):
    """Generate the vendermonde matrix associated with the data set"""
    N = len(x_data)
    A = np.ones((N,N))
    for i in range(1,N):
        A[i] = x_data**i
    return A

def back_substitution(mat_A, b):
    """
    Apply back substitution to an upper triangular matrix
    and return the vector x in the matrix equation Ax = b
    """
    N = len(mat_A[0])
    x = np.zeros(N)
    x[N-1] = b[N-1]/mat_A[N-1,N-1]
    for i in range(2, N):
        sum = 0
        for j in range(i+1, N):
            sum += mat_A[N-i,j]*x[N-j]
        x[N-i] = (b[N-i] - sum)/mat_A[N-i,N-i]
    return x

def forward_substitution(mat_A, b):
    """
    Apply forward substitution to a lower triangular matrix
    and return the vector x in the matrix equation Ax = b
    """
    N = len(mat_A[0])
    x = np.zeros(N)
    x[0] = b[0]/a[0,0]
    for i in range(1, N):
        sum = 0
        for j in range(i-1):
            sum += mat_A[i,j]*x[j]
        x[i] = (b[i] - sum)/mat_A[i,i]
    return x

def find_coeffs_from_data(data):
    """Find the coefficients for a smooth function fitting."""
    A = vandermonde_matrix(data)
    q,r = np.linalg.qr(A)
    coeffs = back_substitution(r, data)
    return coeffs

def polynomial_degree_m(x, coeffs, m):
    """
    Generate p(x) where p is a polynomial of degree mself.
    The polynomial coefficients are found using QR factorization and
    back substitution. (find_coeffs_from_data)
    """
    N = len(x)
    polynomial = np.zeros(N)
    for i in range(N):
        for j in range(m):
            polynomial[i] += coeffs[j]*x[i]**j
    return polynomial

# Datasets as defined from generate_data.py
n = 30
start = -2
stop = 2
x = np.linspace(start, stop, n)
eps = 1
np.random.seed(1) # use same seed every time
r = np.random.random(n) * eps
data_one = x*(np.cos(r + 0.5*x**3) + np.sin(0.5*x**3))
data_two = 4*x**5 - 5*x**4 - 20*x**3 + 10*x*x + 40*x + r

m = 3 # degree of p(x)

# Find coefficients
coeffs_one = find_coeffs_from_data(data_one)
print(coeffs_one)
coeffs_two = find_coeffs_from_data(data_two)

# Plotting
plt.subplot(1,2,1)
plt.plot(x, data_one, 'o')
plt.plot(x, polynomial_degree_m(x, coeffs_one, m), 'r', label='Fit')
plt.legend()

plt.subplot(1,2,2)
plt.plot(x, data_two, 'o')
plt.plot(x, polynomial_degree_m(x, coeffs_two, m), 'r', label='Fit')
plt.legend()

plt.show()

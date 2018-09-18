from generate_data import dataset_one, dataset_two
import numpy as np
import matplotlib.pyplot as plt

# Funciton definitions
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

# Datasets as defined from generate_data.py
data_one = dataset_one()
data_two = dataset_two()

# Matrix operations
A_one = vandermonde_matrix(data_one[0])
q_one,r_one = np.linalg.qr(A_one)
coeff_one = back_substitution(r_one, data_one[1])

A_two = vandermonde_matrix(data_two[0])
q_two,r_two = np.linalg.qr(A_two)
coeff_two = back_substitution(r_two, data_two[1])

# Plotting
plt.subplot(1,2,1)
plt.plot(data_one[0], data_one[1], 'o')
# plt.plot(x_fit_one, polynomial_fit_one, 'r-', title='Fit')
plt.subplot(1,2,2)
plt.plot(data_two[0], data_two[1], 'o')
# plt.plot(x_fit_two, polynomial_fit_two, 'r-', title='Fit')
plt.legend()
plt.show()

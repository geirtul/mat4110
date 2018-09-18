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

def polynomial_degree_m(x, coeffs, m):
    N = len(x)
    p = np.zeros(N)
    for i in range(N):
        for j in range(m):
            p[i] += coeffs[j]*x[i]**(j-1)
    return p


# Datasets as defined from generate_data.py
def dataset_one(x):
    import numpy as np
    eps = 1
    np.random.seed(1) # use same seed every time
    r = np.random.random(n) * eps
    y = x*(np.cos(r + 0.5*x**3) + np.sin(0.5*x**3))
    return y

def dataset_two(x):
    import numpy as np
    eps = 1
    np.random.seed(1) # use same seed every time
    r = np.random.random(n) * eps
    y = 4*x**5 - 5*x**4 - 20*x**3 + 10*x*x + 40*x + r
    return y

n = 30
start = -2
stop = 2
x = np.linspace(start, stop, n)

data_one = dataset_one(x)
data_two = dataset_two(x)

# Matrix operations
A_one = vandermonde_matrix(data_one)
q_one,r_one = np.linalg.qr(A_one)
coeffs_one = back_substitution(r_one, data_one)

A_two = vandermonde_matrix(data_two)
q_two,r_two = np.linalg.qr(A_two)
coeffs_two = back_substitution(r_two, data_two)

# Plotting
plt.subplot(1,2,1)
plt.plot(data_one[0], data_one[1], 'o')
plt.plot(x, polynomial_degree_m(x, coeffs_one, 3), 'r', label='Fit')

plt.subplot(1,2,2)
plt.plot(data_two[0], data_two[1], 'o')
plt.plot(x, polynomial_degree_m(x, coeffs_two, 3), 'r', label='Fit')

plt.legend()
plt.show()

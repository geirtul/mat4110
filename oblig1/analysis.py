import numpy as np
import matplotlib.pyplot as plt

# Function definitions
def vandermonde_matrix(data, m):
    """Generate the vendermonde matrix associated with the data set"""
    N = len(data)
    A = np.zeros((N,m+1))
    A[:,0] = 1.0
    for i in range(1,m+1):
        A[:,i] = data**i
    return A

def back_substitution(mat_A, b):
    """
    Apply back substitution to an upper triangular matrix
    and return the vector x in the matrix equation Ax = b
    """
    N = len(mat_A[0])
    x = np.zeros(N)
    for i in reversed(range(N)):
        sum = 0
        for j in range(i+1,N):
            sum += mat_A[i,j]*x[j]
        x[i] = (b[i] - sum)/mat_A[i,i]
    return x

def forward_substitution(mat_A, b):
    """
    Apply forward substitution to a lower triangular matrix
    and return the vector x in the matrix equation Ax = b
    """
    N = len(mat_A[0])
    x = np.zeros(N)
    for i in range(N):
        sum = 0
        for j in range(i):
            sum += mat_A[i,j]*x[j]
        x[i] = (b[i] - sum)/mat_A[i,i]
    return x

def find_coeffs_from_data(data, m):
    """Find the coefficients for a smooth function fitting."""
    A = vandermonde_matrix(data, m)
    q,r = np.linalg.qr(A)
    coeffs = back_substitution(r, q.T.dot(data))
    return coeffs

def cholesky_factorization(A):
    """Performs cholesky factorization on a positive definite
    matrix A. Here A is defined by vandermonde_matrix()"""
    A_curr = np.copy(A)
    L = np.zeros(A.shape)
    L_T = np.zeros(A.shape)
    D = np.zeros(A.shape)
    for i in range(A.shape[1]):
        L[:,i] = A_curr[:,i]/A_curr[i,i]
        D[i,i] = A_curr[i,i]
        L_dot_L_T = np.c_[L[:,i]].dot(np.c_[L[:,i]].T)
        A_curr = A_curr - D[i,i]*L_dot_L_T

    return [L, D, L.T]


def polynomial_degree_m(x, coeffs, m):
    """
    Generate p(x) where p is a polynomial of degree mself.
    The polynomial coefficients are found using QR factorization and
    back substitution. (find_coeffs_from_data)
    """
    N = len(x)
    polynomial = np.zeros(N)
    for i in range(N):
        for j in range(m+1):
            polynomial[i] += coeffs[j]*x[i]**j
    return polynomial

if __name__ == '__main__':
    # Datasets as defined from generate_data.py
    n = 30
    start = -2.0
    stop = 2.0
    x = np.linspace(start, stop, n)
    eps = 1.0
    np.random.seed(1) # use same seed every time
    r = np.random.random(n) * eps
    data_one = x*(np.cos(r + 0.5*x**3) + np.sin(0.5*x**3))
    data_two = 4*x**5 - 5*x**4 - 20*x**3 + 10*x*x + 40*x + r

    m = 3 # degree of p(x)

    # Find coefficients
    coeffs_one = find_coeffs_from_data(data_one, m)
    #print(coeffs_one)
    #coeffs_two = find_coeffs_from_data(data_two, m)
    #print(coeffs_two)


    # Generate polynomial fits.
    poly_one = polynomial_degree_m(x, coeffs_one, m)
    #poly_two = polynomial_degree_m(x, coeffs_two, m)

    # Plotting
    plt.subplot(1,2,1)
    plt.plot(x, data_one, 'o')
    plt.plot(x, poly_one, 'r', label='Fit')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(x, data_two, 'o')
    #plt.plot(x, poly_two, 'r', label='Fit')
    plt.legend()

    plt.show()

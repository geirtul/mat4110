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
    Apply back substitution to an upper triangular matrix A
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
    Apply forward substitution to a lower triangular matrix A
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

def find_coeffs_from_data(x, data, m):
    """Find the coefficients for a smooth function fitting
    using the QR factorization approach."""
    A = vandermonde_matrix(x, m)
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
    Generate p(x) where p is a polynomial of degree m.
    The polynomial coefficients are found using QR factorization and
    back substitution. (find_coeffs_from_data)
    """
    N = len(x)
    polynomial = np.zeros(N)
    for i in range(N):
        for j in range(m+1):
            polynomial[i] += coeffs[j]*x[i]**j
    return polynomial

def condition_number(A):
    """Find the condition number for a problem given n n x m matrix as
    in Ax = b"""
    u, s, vh = np.linalg.svd(A)
    sigma_max = np.amax(s)
    sigma_min = np.amin(s)
    K = sigma_max / sigma_min
    return K

if __name__ == '__main__':
    """
    Runs the calculations.
    Change 'm' to change the order of the fitted polynomial.
    Change 'eps' to increase/decrese the noise in the datasets.
    """

    # Datasets as defined in task.
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

    # =============================================
    # =  QR-Factorization part                    =
    # =============================================
    # Find coefficients
    coeffs_one = find_coeffs_from_data(x, data_one, m)
    coeffs_two = find_coeffs_from_data(x, data_two, m)
    test_A_one = vandermonde_matrix(x, m)
    test_A_two = vandermonde_matrix(x, m)
    test_coeffs_one = np.linalg.solve(test_A_one.T.dot(test_A_one),
                                        test_A_one.T.dot(data_one))
    test_coeffs_two = np.linalg.solve(test_A_two.T.dot(test_A_two),
                                        test_A_two.T.dot(data_two))

    # Generate polynomial fits.
    poly_one = polynomial_degree_m(x, coeffs_one, m)
    poly_two = polynomial_degree_m(x, coeffs_two, m)

    # =============================================
    # =  Cholesky factorization                   =
    # =============================================

    # Set up matrices
    A_one = vandermonde_matrix(x, m)
    A_one_T = A_one.T
    B_one = A_one.T.dot(A_one)
    L_one,D_one,L_one_T = cholesky_factorization(B_one)

    A_two = vandermonde_matrix(x, m)
    A_two_T = A_two.T
    B_two = A_two.T.dot(A_two)
    L_two,D_two,L_two_T = cholesky_factorization(B_two)

    # Make R and R^T for solving R^T*x=y and Ry=A^T*B
    R_one = L_one.dot(np.sqrt(D_one))
    R_one_T = R_one.T

    R_two = L_two.dot(np.sqrt(D_two))
    R_two_T = R_two.T
    # Solve Ry = A^T B
    coeffs_one_y = forward_substitution(R_one, A_one_T.dot(data_one))
    coeffs_two_y = forward_substitution(R_two, A_two_T.dot(data_two))

    # Solve R_T x = y
    coeffs_one_x = back_substitution(R_one_T, coeffs_one_y)
    coeffs_two_x = back_substitution(R_two_T, coeffs_two_y)

    # Calculating condition numbers.
    A = vandermonde_matrix(x,m)
    B = A.T.dot(A)
    print("Condition number for QR = ", condition_number(A))
    print("Condition number for Cholesky = ", condition_number(B))


    # Plotting
    plt.subplot(1,2,1)
    plt.plot(x, data_one, 'o')
    plt.plot(x, poly_one, 'r', label='QR fit')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(x, data_two, 'o')
    plt.plot(x, poly_two, 'r', label='QR fit')
    plt.suptitle('Least Squares using QR Factorization, m = {}'.format(m))
    plt.legend()

    plt.show()
    #plt.savefig("report/figures/QR_{}.pdf".format(m))

    # Clear axis and figures for saving figures.
    plt.cla()
    plt.clf()

    plt.subplot(1,2,1)
    plt.plot(x, data_one, 'o')
    plt.plot(x, polynomial_degree_m(x, coeffs_one_x, m), 'r', label="Cholesky fit")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(x, data_two, 'o')
    plt.plot(x, polynomial_degree_m(x, coeffs_two_x, m), 'r', label="Cholesky fit")
    plt.suptitle('Least Squares using Cholesky Factorization, m = {}'.format(m))
    plt.legend()

    plt.show()
    #plt.savefig("report/figures/Cholesky_{}.pdf".format(m))

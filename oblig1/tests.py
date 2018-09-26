import numpy as np
import unittest
from analysis import *

# Test variables
x_in = np.array([1,2,3,4])
A_upper = np.c_[[1,0,0], [2,2,0], [3,3,3]]
A_lower = np.c_[[3,3,3], [0,2,2], [0,0,1]]
b_upper = np.array([4,1,3])
b_lower = np.array([3,-1,1])
coeffs = np.array([1,2,3])
m = 2



class TestLinregMethods(unittest.TestCase):

    def test_vandermonde_matrix(self):
        """Test for the vandermonde function"""
        expected_A = np.c_[[1,1,1,1], [1,2,3,4], [1,4,9,16]].astype(float)
        computed_A = vandermonde_matrix(x_in, m)
        self.assertTrue(np.array_equal(expected_A, computed_A))

    def test_back_substitution(self):
        """Test for the back substitution algorithm"""
        expected_x = np.array([3,-1,1], dtype=float)
        computed_x = back_substitution(A_upper, b_upper)
        self.assertTrue(np.array_equal(expected_x, computed_x))

    def test_forward_substitution(self):
        """Test for the back substitution algorithm"""
        expected_x = np.array([1,-2,2], dtype=float)
        computed_x = forward_substitution(A_lower, b_lower)
        self.assertTrue(np.array_equal(expected_x, computed_x))

    def test_polynomial_degree_m(self):
        """Test that the polynomial_degree_m function returns polynomials."""
        expected_p = np.array([6,17,34,57], dtype=float)
        computed_p = polynomial_degree_m(x_in, coeffs, m)
        self.assertTrue(np.array_equal(expected_p, computed_p))

    def test_cholesky_factorization(self):
        """Test that the cholesky factorization algorithm does the thing."""
        testmat = np.c_[[3,4], [4,6]]
        exp_L = np.c_[[1.0,4/3],[0,1.0]]
        exp_D = np.c_[[3.0,0],[0,2/3]]
        exp_L_T = exp_L.T
        comp_L, comp_D, comp_L_T = cholesky_factorization(testmat)

        # Because of floats and unrational numbers np.allclose is used
        # instead of np.array_equal in order to set a suitable threshold
        self.assertTrue(np.allclose(exp_L,comp_L, 1e-16))
        self.assertTrue(np.allclose(exp_D,comp_D,1e-16))
        self.assertTrue(np.allclose(exp_L_T,comp_L_T,1e-16))
if __name__ == '__main__':
    unittest.main()

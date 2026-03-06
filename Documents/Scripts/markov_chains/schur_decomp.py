import numpy as np
from scipy.linalg import schur


def schur_decomposition(P):
    """
    Compute the real Schur decomposition of a transition matrix.
    :param P: Transition matrix (n x n)
    :return: T (quasi-upper triangular Schur form), Z (Schur vectors), eigenvalues
    """
    T, Z = schur(P, output='real')
    eigenvalues = np.linalg.eigvals(P)
    return T, Z, eigenvalues


if __name__ == "__main__":
    P = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5]
    ])
    T, Z, eigenvalues = schur_decomposition(P)
    print("Schur form T:\n", T)
    print("\nSchur vectors Z:\n", Z)
    print("\nEigenvalues:", np.round(eigenvalues, 4))
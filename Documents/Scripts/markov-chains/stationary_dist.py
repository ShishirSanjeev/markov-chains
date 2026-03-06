import numpy as np


def stationary_distribution(Q):
    """
    Compute the stationary distribution of a CTMC from its generator matrix.
    Finds the left eigenvector of Q corresponding to eigenvalue 0 and normalizes it.
    :param Q: Generator matrix (n x n)
    :return: Stationary distribution vector pi
    """
    eigenvalues, eigenvectors = np.linalg.eig(Q.T)
    pi = np.real(eigenvectors[:, np.isclose(eigenvalues, 0)].flatten())
    pi = pi / pi.sum()
    return pi


if __name__ == "__main__":
    Q = np.array([[-20, 20, 0],
                  [12, -32, 20],
                  [0, 12, -12]], dtype=float)
    pi = stationary_distribution(Q)
    print("Stationary distribution:", pi)
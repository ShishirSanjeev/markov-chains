import numpy as np


def decompose_generator(Q):
    """
    Decompose a generator matrix Q into holding rates and the embedded Markov chain.
    :param Q: Generator matrix (n x n)
    :return: rates (1D array of holding rates), S (embedded chain transition matrix)
    """
    rates = -np.diag(Q)
    n = Q.shape[0]
    S = np.zeros_like(Q, dtype=float)

    for i in range(n):
        if rates[i] > 0:
            S[i] = Q[i] / rates[i]
            S[i, i] = 0
        else:
            S[i, i] = 1

    return rates, S


if __name__ == "__main__":
    Q = np.array([[-5, 3, 2],
                  [1, -3, 2],
                  [4, 1, -5]], dtype=float)
    rates, S = decompose_generator(Q)
    print("Holding rates:", rates)
    print("Embedded Markov chain:\n", S)
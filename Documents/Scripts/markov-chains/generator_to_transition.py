import numpy as np
from scipy.linalg import expm


def validate_generator(Q):
    """Check that rows of Q sum to 0."""
    return np.allclose(Q.sum(axis=1), 0)


def validate_transition(P):
    """Check that rows of P sum to 1."""
    return np.allclose(P.sum(axis=1), 1)


def generator_to_transition(Q, t):
    """
    Compute the transition probability matrix P(t) = exp(Q * t).
    :param Q: Generator matrix (n x n)
    :param t: Time point (scalar) or list of time points
    :return: Single P(t) matrix, or list of matrices if t is a list
    """
    if np.isscalar(t):
        return expm(Q * t)
    return [expm(Q * tau) for tau in t]


if __name__ == "__main__":
    Q = np.array([[-0.025, 0.02, 0.005],
                  [0.3, -0.5, 0.2],
                  [0.02, 0.4, -0.42]])

    times = [0.1, 0.9, 1.0]
    if validate_generator(Q):
        for tau in times:
            P_t = generator_to_transition(Q, tau)
            if validate_transition(P_t):
                print(f"Transition matrix for t={tau}:\n{P_t}")
            else:
                print(f"Q does not generate a valid transition matrix at t={tau}.")
    else:
        print("Q is not a valid generator matrix.")
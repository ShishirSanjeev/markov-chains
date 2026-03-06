import numpy as np

def simulate_ctmc(Q, initial_state, max_time):
    """
    Simulate a continuous-time Markov chain.
    :param Q: Generator matrix (n x n)
    :param initial_state: Starting state index
    :param max_time: Simulation end time
    """
    states = [initial_state]
    times = [0.0]
    current = initial_state

    while times[-1] < max_time:
        rate = -Q[current, current]
        if rate == 0:
            break

        wait_time = np.random.exponential(1 / rate)
        times.append(times[-1] + wait_time)

        probs = Q[current].copy()
        probs[current] = 0
        probs = probs / probs.sum()

        current = np.random.choice(len(Q), p=probs)
        states.append(current)

    return times, states


def simulate_dtmc(P, initial_state, steps):
    """
    Simulate a discrete-time Markov chain.
    :param P: Transition matrix (n x n)
    :param initial_state: Starting state index (0-based)
    :param steps: Number of transitions to simulate
    """
    states = [initial_state]
    current = initial_state
    for _ in range(steps):
        current = np.random.choice(len(P), p=P[current])
        states.append(current)
    return states


if __name__ == "__main__":
    generator_matrix = np.array([
        [-0.5, 0.5],
        [0.3, -0.3]
    ])
    times, states = simulate_ctmc(generator_matrix, 0, 10)
    print("CTMC States:", states)
    print("Jump Times:", times)

    transition_matrix = np.array([
        [0.9, 0.1],
        [0.4, 0.6]
    ])
    print("DTMC Path:", simulate_dtmc(transition_matrix, 0, 100))
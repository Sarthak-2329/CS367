# lab6_hopfield.py
# CS367 - Lab 6: Hopfield Network (Associative Memory + Eight-Rook + TSP)

import numpy as np
import matplotlib.pyplot as plt

# Basic Hopfield Network Class

class HopfieldNetwork:
    def __init__(self, size):   # FIXED
        self.size = size
        self.W = np.zeros((size, size))

    def train(self, patterns):
        patterns = np.array(patterns)
        num_patterns, N = patterns.shape
        assert N == self.size

        self.W = np.zeros((N, N))
        for p in patterns:
            self.W += np.outer(p, p)
        np.fill_diagonal(self.W, 0)
        self.W /= num_patterns

    def update_async(self, s, max_iters=50):
        s = s.copy()
        N = self.size
        for _ in range(max_iters):
            old_s = s.copy()
            indices = np.random.permutation(N)
            for i in indices:
                h = np.dot(self.W[i], s)
                s[i] = 1 if h >= 0 else -1
            if np.array_equal(s, old_s):
                break
        return s

    def energy(self, s):
        return -0.5 * s @ self.W @ s


# Part 1: 10x10 Associative Memory + Error Correction

def create_plus_pattern():
    img = -1 * np.ones((10, 10))
    img[4:6, :] = 1
    img[:, 4:6] = 1
    return img.flatten()

def create_square_pattern():
    img = -1 * np.ones((10, 10))
    img[2:8, 2:8] = 1
    img[3:7, 3:7] = -1
    return img.flatten()

def add_noise(pattern, noise_level):
    noisy = pattern.copy()
    N = len(pattern)
    num_flips = int(noise_level * N)
    idx = np.random.choice(N, num_flips, replace=False)
    noisy[idx] *= -1
    return noisy

def experiment_error_correction():
    N = 100
    plus = create_plus_pattern()
    square = create_square_pattern()
    patterns = np.vstack([plus, square])

    hop = HopfieldNetwork(N)
    hop.train(patterns)

    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    print("Noise\tInitialErr\tFinalErr")
    for nl in noise_levels:
        noisy = add_noise(plus, nl)
        rec = hop.update_async(noisy, max_iters=50)
        initial_err = np.sum(noisy != plus)
        final_err = np.sum(rec != plus)
        print(f"{nl:.1f}\t{initial_err}\t\t{final_err}")

    noisy_30 = add_noise(plus, 0.3)
    rec_30 = hop.update_async(noisy_30, max_iters=50)

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].imshow(plus.reshape(10, 10), cmap="gray")
    ax[0].set_title("Original +")
    ax[1].imshow(noisy_30.reshape(10, 10), cmap="gray")
    ax[1].set_title("30% Noise")
    ax[2].imshow(rec_30.reshape(10, 10), cmap="gray")
    ax[2].set_title("Recovered")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig("hopfield_plus_recovery.png")
    plt.close()

# Part 2: Eight-Rook Problem

def eight_rook_hopfield():
    N = 8
    num_neurons = N * N
    A = 1.0
    B = 1.0

    W = np.zeros((num_neurons, num_neurons))
    theta = np.zeros(num_neurons)

    def idx(i, j):
        return i * N + j

    # Row constraints
    for i in range(N):
        for j1 in range(N):
            for j2 in range(N):
                if j1 != j2:
                    n1 = idx(i, j1)
                    n2 = idx(i, j2)
                    W[n1, n2] += -2 * A

    # Column constraints
    for j in range(N):
        for i1 in range(N):
            for i2 in range(N):
                if i1 != i2:
                    n1 = idx(i1, j)
                    n2 = idx(i2, j)
                    W[n1, n2] += -2 * B

    # Thresholds
    for i in range(N):
        for j in range(N):
            n = idx(i, j)
            theta[n] = 2 * A + 2 * B

    np.fill_diagonal(W, 0)

    def energy(x):
        return -0.5 * x @ W @ x + theta @ x

    x = np.random.choice([0, 1], size=num_neurons)

    for _ in range(200):
        n = np.random.randint(num_neurons)
        delta_E = (np.dot(W[n], x) - theta[n])
        x[n] = 1 if delta_E > 0 else 0

    board = x.reshape(N, N)
    print("Eight-Rook Solution (1 = rook):")
    print(board)
    plt.imshow(board, cmap="gray")
    plt.title("Eight-Rook Hopfield Solution")
    plt.savefig("eight_rook_solution.png")
    plt.close()

# Part 3: TSP Hopfield Network

def tsp_hopfield():
    num_cities = 10
    N = num_cities
    num_neurons = N * N

    coords = np.random.rand(N, 2)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    A = 500
    B = 500
    C = 1

    def idx(i, t):
        return i * N + t

    W = np.zeros((num_neurons, num_neurons))
    theta = np.zeros(num_neurons)

    # One city per position
    for i in range(N):
        for t1 in range(N):
            for t2 in range(N):
                if t1 != t2:
                    W[idx(i, t1), idx(i, t2)] += -2 * A

    # One position per city
    for t in range(N):
        for i1 in range(N):
            for i2 in range(N):
                if i1 != i2:
                    W[idx(i1, t), idx(i2, t)] += -2 * B

    # Distance minimization term
    for i in range(N):
        for j in range(N):
            if i != j:
                for t in range(N):
                    t_next = (t + 1) % N
                    W[idx(i, t), idx(j, t_next)] += -C * dist[i, j]

    np.fill_diagonal(W, 0)

    for i in range(N):
        for t in range(N):
            theta[idx(i, t)] = 2 * A + 2 * B

    x = np.random.choice([0, 1], size=num_neurons)

    for _ in range(5000):
        n = np.random.randint(num_neurons)
        delta_E = (np.dot(W[n], x) - theta[n])
        x[n] = 1 if delta_E > 0 else 0

    X = x.reshape(N, N)
    print("TSP Hopfield Assignment Matrix:")
    print(X)

    tour = []
    for t in range(N):
        cities = np.where(X[:, t] == 1)[0]
        if len(cities) == 1:
            tour.append(cities[0])
        else:
            tour.append(-1)
    print("Tour (by position):", tour)

if __name__ == "__main__":   # FIXED
    experiment_error_correction()
    eight_rook_hopfield()
    tsp_hopfield()

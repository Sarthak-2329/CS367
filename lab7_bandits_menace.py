# lab7_bandits_menace.py
# CS367 - Lab 7: MENACE (simple), Binary Bandit, Non-stationary Bandit

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Simple MENACE-like Tic-Tac-Toe Agent

class Menace:
    def __init__(self, initial_beads=3):
        self.matchboxes = defaultdict(lambda: defaultdict(int))
        self.initial_beads = initial_beads

    def board_to_key(self, board):
        return "".join(board)

    def get_moves(self, board):
        return [i for i, v in enumerate(board) if v == " "]

    def choose_move(self, board):
        key = self.board_to_key(board)
        moves = self.get_moves(board)
        beads = np.array([self.matchboxes[key][m] if self.matchboxes[key][m] > 0 else self.initial_beads for m in moves])
        probs = beads / beads.sum()
        move = np.random.choice(moves, p=probs)
        return move

    def update(self, history, result):
        # result: +1 = win, 0 = draw, -1 = loss
        for key, move in history:
            if result == 1:      # reward
                self.matchboxes[key][move] += 1
            elif result == -1:   # punishment
                self.matchboxes[key][move] = max(1, self.matchboxes[key][move] - 1)
            # draw: no change

# Binary Bandit with epsilon-greedy

def run_binary_bandit(true_probs, epsilon=0.1, steps=1000, alpha=0.1):
    n_arms = len(true_probs)
    Q = np.zeros(n_arms)
    rewards = []

    for t in range(steps):
        if np.random.rand() < epsilon:
            a = np.random.randint(n_arms)
        else:
            a = np.argmax(Q)
        r = 1 if np.random.rand() < true_probs[a] else 0
        Q[a] += alpha * (r - Q[a])
        rewards.append(r)
    return np.array(rewards)

def experiment_binary_bandit():
    true_probs = [0.3, 0.7]  # arm 1 worse, arm 2 better
    rewards = run_binary_bandit(true_probs, epsilon=0.1, steps=1000, alpha=0.1)
    avg_reward = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)

    plt.figure()
    plt.plot(avg_reward)
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.title("Binary Bandit: Epsilon-greedy")
    plt.tight_layout()
    plt.savefig("binary_bandit_reward.png")
    plt.close()

# =====================================
# Non-stationary 10-armed bandit
# =====================================

def run_nonstationary_bandit(steps=10000, epsilon=0.1, alpha=None):
    n_arms = 10
    q_true = np.zeros(n_arms)
    Q_sample = np.zeros(n_arms)
    Q_const = np.zeros(n_arms)
    N = np.zeros(n_arms)

    optimal_sample = []
    optimal_const = []

    for t in range(steps):
        q_true += np.random.normal(0, 0.01, size=n_arms)
        optimal_arm = np.argmax(q_true)

        if np.random.rand() < epsilon:
            a_s = np.random.randint(n_arms)
        else:
            a_s = np.argmax(Q_sample)

        if np.random.rand() < epsilon:
            a_c = np.random.randint(n_arms)
        else:
            a_c = np.argmax(Q_const)

        r_s = np.random.normal(q_true[a_s], 1.0)
        r_c = np.random.normal(q_true[a_c], 1.0)

        N[a_s] += 1
        Q_sample[a_s] += (r_s - Q_sample[a_s]) / N[a_s]

        if alpha is None:
            alpha_c = 0.1
        else:
            alpha_c = alpha
        Q_const[a_c] += alpha_c * (r_c - Q_const[a_c])

        optimal_sample.append(1 if a_s == optimal_arm else 0)
        optimal_const.append(1 if a_c == optimal_arm else 0)

    return np.array(optimal_sample), np.array(optimal_const)

def experiment_nonstationary_bandit():
    opt_s, opt_c = run_nonstationary_bandit(steps=10000, epsilon=0.1, alpha=0.1)
    steps = len(opt_s)
    x = np.arange(1, steps + 1)
    avg_s = np.cumsum(opt_s) / x
    avg_c = np.cumsum(opt_c) / x

    plt.figure()
    plt.plot(x, avg_s * 100, label="Sample-average Q")
    plt.plot(x, avg_c * 100, label="Constant step-size Q")
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.title("Non-stationary 10-armed bandit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("nonstationary_bandit.png")
    plt.close()

if __name__ == "__main__":
    experiment_binary_bandit()
    experiment_nonstationary_bandit()

# lab8_gbike.py
# CS367 - Lab 8: Gbike Bicycle Rental (Policy Iteration)

import numpy as np

MAX_BIKES = 20
MAX_MOVE = 5
RENT_REWARD = 10
MOVE_COST = 2
GAMMA = 0.9

POISSON_CUTOFF = 11

from math import exp, factorial

def poisson_prob(lmbda, n):
    return (lmbda**n) * exp(-lmbda) / factorial(n)

# Precompute Poisson pmfs
def build_poisson_cache(lmbda):
    probs = [poisson_prob(lmbda, n) for n in range(POISSON_CUTOFF)]
    tail = 1 - sum(probs)
    probs.append(tail)
    return probs

req1_p = build_poisson_cache(3)
req2_p = build_poisson_cache(4)
ret1_p = build_poisson_cache(3)
ret2_p = build_poisson_cache(2)

def expected_return(state, action, V, modified=False):
    n1, n2 = state

    if action > 0:
        move = min(action, n1, MAX_BIKES - n2)
    else:
        move = -min(-action, n2, MAX_BIKES - n1)

    n1 -= move
    n2 += move

    if modified:
        if move > 0:
            move_cost = MOVE_COST * max(0, move - 1)
        else:
            move_cost = MOVE_COST * abs(move)
    else:
        move_cost = MOVE_COST * abs(move)

    total = 0.0

    for r1 in range(POISSON_CUTOFF + 1):
        for r2 in range(POISSON_CUTOFF + 1):
            p_rent = req1_p[r1] * req2_p[r2]

            real_rent1 = min(n1, r1)
            real_rent2 = min(n2, r2)
            reward = (real_rent1 + real_rent2) * RENT_REWARD

            n1_ = n1 - real_rent1
            n2_ = n2 - real_rent2

            for ret1 in range(POISSON_CUTOFF + 1):
                for ret2 in range(POISSON_CUTOFF + 1):
                    p_return = ret1_p[ret1] * ret2_p[ret2]
                    p = p_rent * p_return

                    n1_final = min(MAX_BIKES, n1_ + ret1)
                    n2_final = min(MAX_BIKES, n2_ + ret2)

                    extra_parking = 0
                    if modified:
                        if n1_final > 10:
                            extra_parking += 4
                        if n2_final > 10:
                            extra_parking += 4

                    total_reward = reward - move_cost - extra_parking
                    total += p * (total_reward + GAMMA * V[n1_final, n2_final])
    return total

def policy_iteration(modified=False):
    V = np.zeros((MAX_BIKES+1, MAX_BIKES+1))
    policy = np.zeros((MAX_BIKES+1, MAX_BIKES+1), dtype=int)

    stable = False
    it = 0
    while not stable:
        it += 1
        print(f"Policy Evaluation Iteration {it}...")
        # Policy evaluation
        for _ in range(5):  # inner sweeps
            new_V = V.copy()
            for n1 in range(MAX_BIKES+1):
                for n2 in range(MAX_BIKES+1):
                    a = policy[n1, n2]
                    new_V[n1, n2] = expected_return((n1, n2), a, V, modified=modified)
            V = new_V

        print("Policy Improvement...")
        stable = True
        for n1 in range(MAX_BIKES+1):
            for n2 in range(MAX_BIKES+1):
                old_a = policy[n1, n2]
                best_val = -1e9
                best_a = 0
                for a in range(-MAX_MOVE, MAX_MOVE+1):
                    if (a > 0 and (a > n1 or a > MAX_BIKES - n2)) or \
                       (a < 0 and (-a > n2 or -a > MAX_BIKES - n1)):
                        continue
                    val = expected_return((n1, n2), a, V, modified=modified)
                    if val > best_val:
                        best_val = val
                        best_a = a
                policy[n1, n2] = best_a
                if best_a != old_a:
                    stable = False

    return V, policy

if __name__ == "__main__":
    print("Running ORIGINAL Gbike problem...")
    V_orig, pi_orig = policy_iteration(modified=False)
    np.save("gbike_V_original.npy", V_orig)
    np.save("gbike_policy_original.npy", pi_orig)

    print("Running MODIFIED Gbike problem...")
    V_mod, pi_mod = policy_iteration(modified=True)
    np.save("gbike_V_modified.npy", V_mod)
    np.save("gbike_policy_modified.npy", pi_mod)

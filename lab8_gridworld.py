# lab8_gridworld.py
# CS367 - Lab 8: 4x3 Gridworld Value Iteration

import numpy as np

# Grid:
# (0,2) (1,2) (2,2) (3,2)
# (0,1) (1,1) (2,1) (3,1)
# (0,0) (1,0) (2,0) (3,0)

# Terminal states: (3,2) with +1, (3,1) with -1
# Blocked state: (1,1)

ACTIONS = ["U", "D", "L", "R"]
ACTION_IDX = {a: i for i, a in enumerate(ACTIONS)}

def in_bounds(s):
    x, y = s
    return 0 <= x <= 3 and 0 <= y <= 2 and not (x == 1 and y == 1)

def move(s, a):
    x, y = s
    if a == "U":
        ns = (x, y+1)
    elif a == "D":
        ns = (x, y-1)
    elif a == "L":
        ns = (x-1, y)
    else:
        ns = (x+1, y)
    if in_bounds(ns):
        return ns
    return s

def transitions(s, a):
    # nondeterministic: 0.8 intended, 0.1 left, 0.1 right
    if s == (3,2) or s == (3,1):
        return [(1.0, s)]
    if a == "U":
        intended = "U"; left = "L"; right = "R"
    elif a == "D":
        intended = "D"; left = "R"; right = "L"
    elif a == "L":
        intended = "L"; left = "D"; right = "U"
    else:
        intended = "R"; left = "U"; right = "D"

    probs = [(0.8, intended), (0.1, left), (0.1, right)]
    result = {}
    for p, act in probs:
        ns = move(s, act)
        result[ns] = result.get(ns, 0) + p
    return [(p, ns) for ns, p in result.items()]

def reward(s, step_cost):
    if s == (3,2):
        return 1.0
    if s == (3,1):
        return -1.0
    return step_cost

def value_iteration(step_cost=-0.04, gamma=0.99, theta=1e-4):
    states = [(x,y) for x in range(4) for y in range(3) if not (x==1 and y==1)]
    V = {s: 0.0 for s in states}
    V[(3,2)] = 1.0
    V[(3,1)] = -1.0

    while True:
        delta = 0
        for s in states:
            if s in [(3,2), (3,1)]:
                continue
            v_old = V[s]
            best = -1e9
            for a in ACTIONS:
                total = 0
                for p, ns in transitions(s, a):
                    r = reward(ns, step_cost)
                    total += p * (r + gamma * V[ns])
                if total > best:
                    best = total
            V[s] = best
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V

def print_values(V):
    for y in reversed(range(3)):
        row = []
        for x in range(4):
            if (x,y) == (1,1):
                row.append("#####")
            else:
                row.append(f"{V[(x,y)]:.2f}")
        print("\t".join(row))
    print()

if __name__ == "__main__":
    for sc in [-2.0, -0.04, 0.02, 1.0]:
        print(f"Step cost r(s) = {sc}")
        V = value_iteration(step_cost=sc, gamma=0.99)
        print_values(V)

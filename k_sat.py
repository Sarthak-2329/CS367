import random

def generate_k_sat(k, m, n):
    """
    Creates a random k-SAT instance.
    Each clause has k literals and we have m total clauses, n vars.
    """
    if k > n:
        raise ValueError("k > n makes no sense")

    vars_list = list(range(1, n + 1))
    sat_problem = []

    for _ in range(m):
        clause = random.sample(vars_list, k)
        for i in range(k):
            if random.choice([True, False]):
                clause[i] = -clause[i]
        sat_problem.append(set(clause))

    return sat_problem


def evaluate(state, problem):
    """Counts how many clauses are satisfied under the given assignment."""
    satisfied = 0
    for clause in problem:
        for literal in clause:
            idx = abs(literal) - 1
            if (literal > 0 and state[idx] == 1) or (literal < 0 and state[idx] == 0):
                satisfied += 1
                break
    return satisfied


def hill_climbing(problem, n, max_iters=1000):
    """
    Simplest form of hill climbing for SAT.
    Just keeps flipping bits hoping to improve score.
    """
    curr_state = [random.randint(0, 1) for _ in range(n)]
    curr_score = evaluate(curr_state, problem)

    for step in range(max_iters):
        if curr_score == len(problem):
            return curr_state

        best_state = None
        best_score = curr_score

        for i in range(n):
            temp = curr_state[:]
            temp[i] = 1 - temp[i]
            sc = evaluate(temp, problem)
            if sc > best_score:
                best_score = sc
                best_state = temp

        if best_state is None:
            return None

        curr_state = best_state
        curr_score = best_score

    return None


def beam_search(problem, n, beam_width=4, max_steps=100):
    """
    Beam search with random initial states.
    Retains top 'beam_width' candidates per step.
    """
    beam = [[random.randint(0, 1) for _ in range(n)] for _ in range(beam_width)]

    for t in range(max_steps):
        new_candidates = []
        for s in beam:
            if evaluate(s, problem) == len(problem):
                return s

            for i in range(n):
                neighbor = s[:]
                neighbor[i] = 1 - neighbor[i]
                new_candidates.append(neighbor)

        if not new_candidates:
            return None

        scored = sorted(new_candidates, key=lambda x: evaluate(x, problem), reverse=True)
        beam = scored[:beam_width]

    return None


def get_vnd_neighbors(state, neighborhood_size):
    """Flip 'neighborhood_size' random bits to make a neighbor."""
    n = len(state)
    indices = random.sample(range(n), neighborhood_size)
    neighbor = state[:]
    for idx in indices:
        neighbor[idx] = 1 - neighbor[idx]
    return [neighbor]


def variable_neighborhood_descent(problem, n, max_neighborhoods=3, max_steps=1000):
    """
    Variable Neighborhood Descent for SAT.
    Starts small, explores larger neighborhoods if stuck.
    """
    curr_state = [random.randint(0, 1) for _ in range(n)]

    for _ in range(max_steps):
        score = evaluate(curr_state, problem)
        if score == len(problem):
            return curr_state

        k = 1
        while k <= max_neighborhoods:
            neighbor = get_vnd_neighbors(curr_state, k)[0]
            neigh_score = evaluate(neighbor, problem)

            if neigh_score > score:
                curr_state = neighbor
                k = 1
            else:
                k += 1
    return None


def run_experiment(algorithm, problem_params, num_instances=20):
    """
    Repeats an experiment for several random problem instances.
    Calculates % of runs that found a satisfying assignment.
    """
    k, m, n = problem_params
    solved = 0

    for i in range(num_instances):
        print(f"  -> Running instance {i + 1}/{num_instances}", end="\r")
        sat_instance = generate_k_sat(k, m, n)
        sol = None

        if algorithm == 'hill_climbing':
            sol = hill_climbing(sat_instance, n)
        elif algorithm == 'beam_search':
            sol = beam_search(sat_instance, n)
        elif algorithm == 'vnd':
            sol = variable_neighborhood_descent(sat_instance, n)

        if sol is not None and evaluate(sol, sat_instance) == len(sat_instance):
            solved += 1

    print()
    return (solved / num_instances) * 100


if __name__ == "__main__":
    k = 3
    configs = {
        "Hill Climbing": [(10, 10), (25, 25), (50, 50)],
        "Beam Search": [(5, 5), (10, 10), (25, 25)],
        "VND": [(10, 10), (25, 25), (50, 50)]
    }

    results = {}

    for algo_name, setups in configs.items():
        print(f"\n--- Testing {algo_name} ---")
        results[algo_name] = {}
        func_name = algo_name.lower().replace(" ", "_")

        for m, n in setups:
            params = (k, m, n)
            print(f"Running config: m={m}, n={n}")
            rate = run_experiment(func_name, params)
            results[algo_name][(m, n)] = rate
            print(f"Penetrance: {rate:.2f}%")
        print("-" * 25)

    print("\n=== Summary ===")
    for algo, vals in results.items():
        print(f"\nAlgorithm: {algo}")
        for conf, score in vals.items():
            print(f"  Config {conf}: {score:.1f}% success rate")

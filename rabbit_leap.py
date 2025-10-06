from collections import deque

def solve_with_bfs(start_state, goal_state):
    """Breadth-first search (should guarantee shortest path)."""
    to_visit = deque([(start_state, [start_state])])
    seen = {start_state}

    max_q_len = 1
    nodes_checked = 0

    while to_visit:
        nodes_checked += 1
        current, path = to_visit.popleft()

        if current == goal_state:
            return path, nodes_checked, max_q_len

        next_states = get_successors(current)
        for nxt in next_states:
            if nxt not in seen:
                seen.add(nxt)
                to_visit.append((nxt, path + [nxt]))
                if len(to_visit) > max_q_len:
                    max_q_len = len(to_visit)

    return None, nodes_checked, max_q_len


def solve_with_dfs(start_state, goal_state):
    """Depth-first search (not always optimal, but quicker sometimes)."""
    pending = [(start_state, [start_state])]
    visited = set()
    biggest_stack = 1
    count = 0

    while pending:
        count += 1
        current_state, path = pending.pop()

        if current_state == goal_state:
            return path, count, biggest_stack

        if current_state in visited:
            continue

        visited.add(current_state)

        for nxt in reversed(get_successors(current_state)):
            if nxt not in visited:
                new_path = path + [nxt]
                pending.append((nxt, new_path))
                if len(pending) > biggest_stack:
                    biggest_stack = len(pending)

    return None, count, biggest_stack


def print_solution(name, path, visited, max_size):
    """Just prints the results — not fancy, but works."""
    print(f"\n=== {name} Results ===")
    if path:
        print("Solution found!\n")
        for idx, step in enumerate(path):
            print(f"Step {idx}: {step}")
        print("\n-- Stats --")
        print(f"Visited Nodes: {visited}")
        print(f"Peak Queue/Stack Size: {max_size}")
        print(f"Solution Path Length: {len(path)}")
    else:
        print("No solution found :(")
    print("-" * 30)


def get_successors(state):
    """
    Returns a list of possible next moves.
    (Might refactor later if this grows messy.)
    """
    options = []
    hole = state.find('_')

    if hole > 0 and state[hole - 1] == 'E':
        s = list(state)
        s[hole], s[hole - 1] = s[hole - 1], s[hole]
        options.append(''.join(s))

    if hole > 1 and state[hole - 2] == 'E' and state[hole - 1] == 'W':
        temp = list(state)
        temp[hole], temp[hole - 2] = temp[hole - 2], temp[hole]
        options.append(''.join(temp))

    if hole < len(state) - 1 and state[hole + 1] == 'W':
        s = list(state)
        s[hole], s[hole + 1] = s[hole + 1], s[hole]
        options.append(''.join(s))

    if hole < len(state) - 2 and state[hole + 2] == 'W' and state[hole + 1] == 'E':
        s = list(state)
        s[hole], s[hole + 2] = s[hole + 2], s[hole]
        options.append(''.join(s))

    return options


if __name__ == "__main__":
    start = "EEE_WWW"
    goal = "WWW_EEE"

    print(f"Starting Rabbit Leap puzzle: {start} → {goal}\n")

    bfs_path, bfs_nodes, bfs_max = solve_with_bfs(start, goal)
    print_solution("BFS", bfs_path, bfs_nodes, bfs_max)

    dfs_path, dfs_nodes, dfs_max = solve_with_dfs(start, goal)
    print_solution("DFS", dfs_path, dfs_nodes, dfs_max)

    print("\n=== Comparison Summary ===")
    if bfs_path and dfs_path:
        print(f"BFS: {len(bfs_path)-1} moves (optimal)")
        print(f"DFS: {len(dfs_path)-1} moves (might be longer)")
        print("\nBFS guarantees shortest path, DFS explores deeper paths first.")
        print("I might visualize this later... (future idea!)")
    else:
        print("Couldn't find both solutions to compare.")

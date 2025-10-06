import heapq
import spacy
import string

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("⚠️  SpaCy model not found. Try running:")
    print("    python -m spacy download en_core_web_sm")
    exit(1)

def preprocess_text(text):
    """Lowercases + removes punctuation + splits into sentences using SpaCy."""
    doc = nlp(text.lower())
    results = []
    for sent in doc.sents:
        cleaned = sent.text.translate(str.maketrans('', '', string.punctuation)).strip()
        if cleaned:
            results.append(cleaned)
    return results


def levenshtein_distance(a, b):
    """Computes Levenshtein distance manually (kinda slow, but fine)."""
    if len(a) < len(b):
        return levenshtein_distance(b, a)
    if not b:
        return len(a)

    prev_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a):
        curr = [i + 1]
        for j, char_b in enumerate(b):
            ins = prev_row[j + 1] + 1
            dele = curr[j] + 1
            sub = prev_row[j] + (char_a != char_b)
            curr.append(min(ins, dele, sub))
        prev_row = curr
    return prev_row[-1]


def heuristic(state, sents1, sents2):
    """
    Just returns difference in remaining sentences.
    TODO: maybe use average Levenshtein later for tighter bound.
    """
    i, j = state
    rem1 = len(sents1) - i
    rem2 = len(sents2) - j
    return abs(rem1 - rem2)


def a_star_align(doc1_text, doc2_text):
    """Aligns two docs sentence-by-sentence using A* search."""
    sents1 = preprocess_text(doc1_text)
    sents2 = preprocess_text(doc2_text)

    start = (0, 0)
    goal = (len(sents1), len(sents2))

    frontier = [(0, 0, start, [])]
    seen_states = set()

    while frontier:
        f, g, current, path = heapq.heappop(frontier)

        if current == goal:
            return path, g

        if current in seen_states:
            continue
        seen_states.add(current)

        i, j = current
        next_moves = []

        if i < len(sents1) and j < len(sents2):
            next_moves.append(((i + 1, j + 1), 'align'))
        if i < len(sents1):
            next_moves.append(((i + 1, j), 'delete'))
        if j < len(sents2):
            next_moves.append(((i, j + 1), 'insert'))

        for nxt, action in next_moves:
            if nxt in seen_states:
                continue

            if action == 'align':
                step_cost = levenshtein_distance(sents1[i], sents2[j])
            elif action == 'delete':
                step_cost = len(sents1[i])
            else:
                step_cost = len(sents2[j])

            new_g = g + step_cost
            new_f = new_g + heuristic(nxt, sents1, sents2)
            heapq.heappush(frontier, (new_f, new_g, nxt, path + [action]))


    return None, float('inf')


if __name__ == "__main__":
    cases = {
        "Identical Docs": (
            "The quick brown fox jumps over the lazy dog. This is a test.",
            "The quick brown fox jumps over the lazy dog. This is a test."
        ),
        "Slightly Modified": (
            "The quick brown fox jumps over the lazy dog. This is a test for plagiarism.",
            "The fast brown fox leaped over the tired dog. This is a check for plagiarism."
        ),
        "Completely Different": (
            "Artificial intelligence is a field of computer science. It focuses on creating intelligent machines.",
            "Photosynthesis is a process used by plants. It converts light energy into chemical energy."
        ),
        "Partial Overlap": (
            "The team won the championship game. It was a historic victory. The weather was perfect for the match.",
            "The weather was perfect for the match. After that, the team celebrated their win. It was a historic victory."
        ),
    }

    for name, (t1, t2) in cases.items():
        print(f"\n--- Running {name} ---")
        path, total_cost = a_star_align(t1, t2)
        if path:
            print(f"✅ Alignment found! Total cost: {total_cost}")
        else:
            print("❌ No alignment found.")
        print("-" * 50)

"""
rule_solver.py 

The core "rule-induction engine". Given 1-2 solved examples of a puzzle,
it searches a defined space of arithmetic operator combinations, finds one
that is consistent with ALL given examples, and applies it to the unsolved
question -- returning both the answer and a human-readable explanation.

This intentionally mirrors (a superset of) the rule templates used in
generate_dataset.py, so it can solve both generated puzzles and, ideally,
real exam-style puzzles that follow similar logic.
"""

import itertools

# Candidate operator functions for 3-input puzzles (triangle / grid)
CANDIDATE_RULES_3INPUT = [
    ("a + b + c",             lambda a, b, c: a + b + c),
    ("(a + b + c) * 2",       lambda a, b, c: (a + b + c) * 2),
    ("a * b - c",             lambda a, b, c: a * b - c),
    ("a + b - c",             lambda a, b, c: a + b - c),
    ("(a + b) - (c * 2)",     lambda a, b, c: (a + b) - (c * 2)),
    ("a * c - b",             lambda a, b, c: a * c - b),
    ("(a + c) * b // 2",      lambda a, b, c: ((a + c) * b) // 2 if b % 2 == 0 or (a + c) % 2 == 0 else None),
    ("a^2 - b - c",           lambda a, b, c: a * a - b - c),
    ("((a+b+c)//3)*5",        lambda a, b, c: ((a + b + c) // 3) * 5),
    ("a + b*2 - c",           lambda a, b, c: a + b * 2 - c),
    ("a - b + c",             lambda a, b, c: a - b + c),
    ("b + c - a",             lambda a, b, c: b + c - a),
    ("a * b + c",             lambda a, b, c: a * b + c),
    ("a * b * c",             lambda a, b, c: a * b * c),
    ("(a + b) * c",           lambda a, b, c: (a + b) * c),
]

def induce_rule_3input(examples):
    """
    examples: list of dicts like {"corners": [a,b,c], "center": value}
              or {"grid": [a,b,c], "missing": value}
    Returns (formula_string, fn) of the first rule consistent with ALL examples,
    or (None, None) if nothing in the candidate space matches.
    """
    for formula, fn in CANDIDATE_RULES_3INPUT:
        try:
            consistent = True
            for ex in examples:
                nums = ex.get("corners") or ex.get("grid")
                target = ex.get("center")
                if target is None:
                    target = ex.get("missing")
                a, b, c = nums
                result = fn(a, b, c)
                if result != target:
                    consistent = False
                    break
            if consistent:
                return formula, fn
        except (TypeError, ZeroDivisionError):
            continue
    return None, None


def solve_3input_puzzle(examples, question_nums):
    """
    Main entry point for triangle/grid puzzles.
    Returns dict with answer, matched formula, and step explanation.
    """
    formula, fn = induce_rule_3input(examples)
    if fn is None:
        return {
            "success": False,
            "message": "Could not find a consistent rule in the current search space. "
                       "Try adding another solved example, or the rule may use an "
                       "operator combination outside the current candidate set."
        }

    a, b, c = question_nums
    answer = fn(a, b, c)
    return {
        "success": True,
        "formula": formula,
        "answer": answer,
        "explanation": (
            f"Verified rule '{formula}' against all given examples -> consistent. "
            f"Applying it to the new numbers ({a}, {b}, {c}) gives {answer}."
        )
    }


# ---------------------------------------------------------------------------
# Number series solver: detects arithmetic, geometric, square-offset,
# and second-difference patterns.
# ---------------------------------------------------------------------------

def _try_alternating_pattern(sequence):
    """
    Detects the 'alternating +k/-j' family used by generate_dataset.py's S4 rule:
        term(i) = start + (k if i is even else -(k // 2)) * i
    This is NOT a constant-difference or constant-ratio pattern, so it needs its
    own brute-force search over k rather than the generic difference checks.
    """
    if len(sequence) < 3:
        return None

    start = sequence[0]
    for k in range(1, 60):
        predicted = [start + (k if i % 2 == 0 else -k // 2) * i for i in range(len(sequence) + 1)]
        if predicted[:len(sequence)] == list(sequence):
            next_term = predicted[len(sequence)]
            return {
                "success": True,
                "formula": f"alternating pattern, start={start}, k={k} "
                           f"(even positions: +{k}*position, odd positions: {-k // 2}*position)",
                "answer": next_term,
                "explanation": (
                    f"This sequence alternates between two different step rules depending on "
                    f"position: even-indexed terms follow start + {k}*position, odd-indexed terms "
                    f"follow start + ({-k // 2})*position. Verified against all given terms, "
                    f"the next term is {next_term}."
                )
            }
    return None


def solve_series(sequence):
    diffs = [sequence[i + 1] - sequence[i] for i in range(len(sequence) - 1)]

    # Arithmetic: constant difference
    if len(set(diffs)) == 1:
        k = diffs[0]
        answer = sequence[-1] + k
        return {"success": True, "formula": f"arithmetic, common difference {k}",
                "answer": answer, "explanation": f"Each term increases by {k}. Next term = {sequence[-1]} + {k} = {answer}."}

    # Geometric: constant ratio
    ratios = [sequence[i + 1] / sequence[i] for i in range(len(sequence) - 1) if sequence[i] != 0]
    if ratios and all(abs(r - ratios[0]) < 1e-6 for r in ratios):
        k = ratios[0]
        answer = int(round(sequence[-1] * k))
        return {"success": True, "formula": f"geometric, common ratio {k}",
                "answer": answer, "explanation": f"Each term is multiplied by {k}. Next term = {sequence[-1]} * {k} = {answer}."}

    # Second difference constant (covers square-offset and difference-of-differences series)
    second_diffs = [diffs[i + 1] - diffs[i] for i in range(len(diffs) - 1)]
    if second_diffs and len(set(second_diffs)) == 1:
        d2 = second_diffs[0]
        next_diff = diffs[-1] + d2
        answer = sequence[-1] + next_diff
        return {"success": True, "formula": f"second difference constant ({d2})",
                "answer": answer, "explanation": f"The difference between terms itself increases by {d2} each time. "
                                                   f"Next difference = {diffs[-1]} + {d2} = {next_diff}, "
                                                   f"so next term = {sequence[-1]} + {next_diff} = {answer}."}

    # Alternating +k/-j pattern (checked last since it needs a parameter search)
    alt_result = _try_alternating_pattern(sequence)
    if alt_result:
        return alt_result

    return {"success": False, "message": "No matching pattern found in the current search space "
                                          "(tried arithmetic, geometric, second-difference, and alternating rules)."}


if __name__ == "__main__":
    # Quick smoke test
    ex = [{"corners": [18, 7, 11], "center": 36}, {"corners": [7, 9, 17], "center": 33}]
    print(solve_3input_puzzle(ex, [8, 16, 10]))
    print(solve_series([10, 13, 19, 28, 40]))

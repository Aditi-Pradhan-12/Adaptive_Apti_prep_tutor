"""
generate_dataset.py

Generates a large, perfectly-labeled synthetic dataset of numeric pattern-reasoning
puzzles (triangle/circle number puzzles, grid puzzles, and number series) for the
Adaptive Aptitude Prep Tutor project.

Because these puzzle types are rule-based, we can generate unlimited examples with
guaranteed-correct answers by randomly sampling numbers and applying a known rule
template. This script is the source of the project's question bank + validation set.

Run:
    python generate_dataset.py --rows 75000 --out dataset.csv
"""

import argparse
import json
import random
import uuid

# ---------------------------------------------------------------------------
# RULE TEMPLATES
# Each template is a small function of the "given" numbers -> the missing number.
# These are the same style of rules real aptitude tests use.
# ---------------------------------------------------------------------------

TRIANGLE_RULES = [
    {"id": "T1", "formula": "center = a + b + c",            "fn": lambda a, b, c: a + b + c},
    {"id": "T2", "formula": "center = (a + b + c) * 2",       "fn": lambda a, b, c: (a + b + c) * 2},
    {"id": "T3", "formula": "center = a * b - c",             "fn": lambda a, b, c: a * b - c},
    {"id": "T4", "formula": "center = a + b - c",             "fn": lambda a, b, c: a + b - c},
    {"id": "T5", "formula": "center = (a + b) - (c * 2)",     "fn": lambda a, b, c: (a + b) - (c * 2)},
    {"id": "T6", "formula": "center = a * c - b",             "fn": lambda a, b, c: a * c - b},
    {"id": "T7", "formula": "center = (a + c) * b // 2",      "fn": lambda a, b, c: ((a + c) * b) // 2},
    {"id": "T8", "formula": "center = a^2 - b - c",           "fn": lambda a, b, c: a * a - b - c},
    {"id": "T9", "formula": "center = (a + b + c) // 3 * 5",  "fn": lambda a, b, c: ((a + b + c) // 3) * 5},
    {"id": "T10","formula": "center = a + b*2 - c",           "fn": lambda a, b, c: a + b * 2 - c},
]

GRID_RULES = [
    # 2x2 grid: a b / c d  where d = f(a,b,c)
    {"id": "G1", "formula": "d = a + b + c",                  "fn": lambda a, b, c: a + b + c},
    {"id": "G2", "formula": "d = (a + b) - c",                "fn": lambda a, b, c: (a + b) - c},
    {"id": "G3", "formula": "d = a * b // c if c != 0 else a * b", "fn": lambda a, b, c: (a * b) // c if c != 0 else a * b},
    {"id": "G4", "formula": "d = a + c - b",                  "fn": lambda a, b, c: a + c - b},
    {"id": "G5", "formula": "d = (a * c) - (b * 2)",          "fn": lambda a, b, c: (a * c) - (b * 2)},
]

SERIES_RULES = [
    {"id": "S1", "formula": "arithmetic (+k)",         "fn": lambda start, k, n: [start + k * i for i in range(n)]},
    {"id": "S2", "formula": "geometric (*k)",           "fn": lambda start, k, n: [start * (k ** i) for i in range(n)]},
    {"id": "S3", "formula": "squares offset",           "fn": lambda start, k, n: [start + (i + 1) ** 2 for i in range(n)]},
    {"id": "S4", "formula": "alternating +k/-j",        "fn": lambda start, k, n: [start + (k if i % 2 == 0 else -k // 2) * i for i in range(n)]},
    {"id": "S5", "formula": "difference-of-differences","fn": lambda start, k, n: [start + k * i * (i + 1) // 2 for i in range(n)]},
]

DIFFICULTY_BY_RULE_COMPLEXITY = {
    "T1": "easy", "T2": "easy", "T3": "medium", "T4": "easy", "T5": "medium",
    "T6": "medium", "T7": "hard", "T8": "hard", "T9": "hard", "T10": "medium",
    "G1": "easy", "G2": "easy", "G3": "medium", "G4": "medium", "G5": "hard",
    "S1": "easy", "S2": "medium", "S3": "medium", "S4": "hard", "S5": "hard",
}


def gen_triangle_row():
    rule = random.choice(TRIANGLE_RULES)
    # Two fully solved example triangles + one incomplete (the "question")
    examples = []
    for _ in range(2):
        a, b, c = random.randint(2, 20), random.randint(2, 20), random.randint(2, 20)
        center = rule["fn"](a, b, c)
        examples.append({"corners": [a, b, c], "center": center})

    qa, qb, qc = random.randint(2, 20), random.randint(2, 20), random.randint(2, 20)
    answer = rule["fn"](qa, qb, qc)

    return {
        "id": str(uuid.uuid4())[:8],
        "puzzle_type": "triangle",
        "rule_id": rule["id"],
        "rule_formula": rule["formula"],
        "difficulty": DIFFICULTY_BY_RULE_COMPLEXITY[rule["id"]],
        "given_examples": json.dumps(examples),
        "question": json.dumps({"corners": [qa, qb, qc], "center": None}),
        "correct_answer": answer,
    }


def gen_grid_row():
    rule = random.choice(GRID_RULES)
    examples = []
    for _ in range(2):
        a, b, c = random.randint(2, 15), random.randint(2, 15), random.randint(1, 15)
        d = rule["fn"](a, b, c)
        examples.append({"grid": [a, b, c], "missing": d})

    qa, qb, qc = random.randint(2, 15), random.randint(2, 15), random.randint(1, 15)
    answer = rule["fn"](qa, qb, qc)

    return {
        "id": str(uuid.uuid4())[:8],
        "puzzle_type": "grid",
        "rule_id": rule["id"],
        "rule_formula": rule["formula"],
        "difficulty": DIFFICULTY_BY_RULE_COMPLEXITY[rule["id"]],
        "given_examples": json.dumps(examples),
        "question": json.dumps({"grid": [qa, qb, qc], "missing": None}),
        "correct_answer": answer,
    }


def gen_series_row():
    rule = random.choice(SERIES_RULES)
    start = random.randint(1, 10)
    k = random.randint(2, 5)
    n = 6  # sequence length shown, last one is the "question"
    full_seq = rule["fn"](start, k, n)
    shown = full_seq[:-1]
    answer = full_seq[-1]

    return {
        "id": str(uuid.uuid4())[:8],
        "puzzle_type": "series",
        "rule_id": rule["id"],
        "rule_formula": rule["formula"],
        "difficulty": DIFFICULTY_BY_RULE_COMPLEXITY[rule["id"]],
        "given_examples": json.dumps({"start": start, "k": k}),
        "question": json.dumps({"sequence": shown, "next": None}),
        "correct_answer": answer,
    }


GENERATORS = [gen_triangle_row, gen_grid_row, gen_series_row]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=75000)
    parser.add_argument("--out", type=str, default="dataset.csv")
    args = parser.parse_args()

    import csv
    fieldnames = ["id", "puzzle_type", "rule_id", "rule_formula", "difficulty",
                  "given_examples", "question", "correct_answer"]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(args.rows):
            gen_fn = GENERATORS[i % 3] if i < args.rows - args.rows % 3 else random.choice(GENERATORS)
            writer.writerow(gen_fn())

    print(f"Generated {args.rows} rows -> {args.out}")


if __name__ == "__main__":
    main()

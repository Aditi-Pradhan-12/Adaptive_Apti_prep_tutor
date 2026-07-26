"""
verbal_dataset_generator.py

Generates synthetic verbal reasoning questions across four sub-types
(matching proposal section 5.4): blood relations, direction sense,
coding-decoding, and syllogisms. Also doubles as labeled training data
for the sub-type classifier in verbal_classifier.py.

Run:
    python verbal_dataset_generator.py --rows 4000 --out verbal_dataset.csv
"""

import argparse
import csv
import random

NAMES = ["Amit", "Riya", "Karan", "Sneha", "Vikram", "Anjali", "Rohit", "Priya",
         "Suresh", "Meera", "Arjun", "Kavya", "Nikhil", "Pooja", "Manoj", "Divya"]

DIRECTIONS = ["north", "south", "east", "west"]
DIR_VECTORS = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}

# --- Blood relation composition table.
# Semantics: "X is Y's REL" means X occupies role REL relative to Y
# (e.g. "A is B's father" = A is the father of B = A is one generation ABOVE B).
#
# To keep every generated answer verifiably correct, we ONLY compose relations
# that move consistently in one direction across both statements:
#   - both ascending (father/mother -> father/mother) => grandparent
#   - both descending (son/daughter -> son/daughter)   => grandchild
# Mixed directions (e.g. son + father) land back on the SAME shared relative
# (e.g. a sibling or a spouse) rather than a further-removed relative, and
# correctly resolving which one needs branch-tracking beyond a simple lookup
# table -- so those combinations are deliberately excluded here rather than
# risk generating an incorrect answer.
BLOOD_COMPOSITIONS = {
    ("father", "father"): "grandfather", ("father", "mother"): "grandfather",
    ("mother", "father"): "grandmother", ("mother", "mother"): "grandmother",
    ("son", "son"): "grandson", ("son", "daughter"): "grandson",
    ("daughter", "son"): "granddaughter", ("daughter", "daughter"): "granddaughter",
}
ASCENDING = {"father", "mother"}
DESCENDING = {"son", "daughter"}


def gen_blood_relation_row():
    a, b, c = random.sample(NAMES, 3)
    rel1 = random.choice(["father", "mother", "son", "daughter"])
    # Force rel2 to move in the SAME direction as rel1 (both ascending or both
    # descending), so the composition is unambiguous and verifiably correct.
    if rel1 in ASCENDING:
        rel2 = random.choice(list(ASCENDING))
    else:
        rel2 = random.choice(list(DESCENDING))
    answer = BLOOD_COMPOSITIONS[(rel1, rel2)]

    question = (f"{a} is {b}'s {rel1}. {b} is {c}'s {rel2}. "
                f"How is {a} related to {c}?")
    return {
        "subtype": "blood_relation",
        "question": question,
        "structured": f"{a}|{rel1}|{b};{b}|{rel2}|{c}",
        "correct_answer": answer,
    }


def gen_direction_row():
    name = random.choice(NAMES)
    n_moves = random.randint(2, 3)
    moves = []
    x, y = 0, 0
    for _ in range(n_moves):
        d = random.choice(DIRECTIONS)
        dist = random.randint(2, 10)
        dx, dy = DIR_VECTORS[d]
        x += dx * dist
        y += dy * dist
        moves.append(f"{dist}km {d}")

    distance = round((x ** 2 + y ** 2) ** 0.5, 2)
    question = f"{name} walks " + ", then ".join(moves) + ". How far is {} from the starting point?".format(name)
    return {
        "subtype": "direction_sense",
        "question": question,
        "structured": ";".join(moves),
        "correct_answer": distance,
    }


def gen_coding_row():
    word = random.choice(["CAT", "DOG", "SUN", "MAP", "PEN", "BAT", "CUP", "FAN"])
    shift = random.randint(1, 5)
    coded = "".join(chr((ord(ch) - 65 + shift) % 26 + 65) for ch in word)

    new_word = random.choice(["BAT", "COW", "TOP", "RUN", "BOX", "JAM", "KEY", "LOG"])
    new_coded = "".join(chr((ord(ch) - 65 + shift) % 26 + 65) for ch in new_word)

    question = f"If {word} is coded as {coded}, how is {new_word} coded?"
    return {
        "subtype": "coding_decoding",
        "question": question,
        "structured": f"{word}->{coded};{new_word}->?",
        "correct_answer": new_coded,
    }


SYLLOGISM_FORMS = [
    {"p1": "All {A} are {B}", "p2": "All {B} are {C}", "concl": "All {A} are {C}", "valid": True, "name": "Barbara"},
    {"p1": "All {A} are {B}", "p2": "No {B} are {C}", "concl": "No {A} are {C}", "valid": True, "name": "Celarent"},
    {"p1": "Some {A} are {B}", "p2": "All {B} are {C}", "concl": "Some {A} are {C}", "valid": True, "name": "Darii"},
    {"p1": "No {A} are {B}", "p2": "Some {B} are {C}", "concl": "Some {C} are not {A}", "valid": True, "name": "Ferio"},
    # Invalid form for negative examples
    {"p1": "Some {A} are {B}", "p2": "Some {B} are {C}", "concl": "Some {A} are {C}", "valid": False, "name": "Invalid"},
]

CATEGORIES = ["cats", "dogs", "animals", "mammals", "pets", "birds", "flowers",
              "plants", "vehicles", "cars", "fruits", "foods"]


def gen_syllogism_row():
    form = random.choice(SYLLOGISM_FORMS)
    a, b, c = random.sample(CATEGORIES, 3)
    p1 = form["p1"].format(A=a, B=b, C=c)
    p2 = form["p2"].format(A=a, B=b, C=c)
    concl = form["concl"].format(A=a, B=b, C=c)

    question = f"Statement 1: {p1}. Statement 2: {p2}. Conclusion: {concl}. Is the conclusion valid?"
    return {
        "subtype": "syllogism",
        "question": question,
        "structured": f"{p1}|{p2}|{concl}",
        "correct_answer": "valid" if form["valid"] else "invalid",
    }


GENERATORS = [gen_blood_relation_row, gen_direction_row, gen_coding_row, gen_syllogism_row]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--out", type=str, default="verbal_dataset.csv")
    args = parser.parse_args()

    fieldnames = ["subtype", "question", "structured", "correct_answer"]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(args.rows):
            gen_fn = GENERATORS[i % 4] if i < args.rows - args.rows % 4 else random.choice(GENERATORS)
            writer.writerow(gen_fn())

    print(f"Generated {args.rows} rows -> {args.out}")


if __name__ == "__main__":
    main()

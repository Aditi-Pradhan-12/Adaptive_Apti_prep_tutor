"""
verbal_solver.py

Logic-based solvers for the four verbal reasoning sub-types (proposal
section 5.4). These are deliberately NOT ML/DL -- they're structured
parsing + rule application, the same design philosophy as rule_solver.py
for the quantitative puzzles: deterministic, verifiable, explainable.

solve_verbal_question() is the top-level router: it uses the trained
classifier (verbal_classifier.py) to detect sub-type, then dispatches
to the matching solver below.
"""

import re
import math

from verbal_classifier import classify_subtype

# ---------------------------------------------------------------------------
# Blood relations
# ---------------------------------------------------------------------------

BLOOD_COMPOSITIONS = {
    ("father", "father"): "grandfather", ("father", "mother"): "grandfather",
    ("mother", "father"): "grandmother", ("mother", "mother"): "grandmother",
    ("son", "son"): "grandson", ("son", "daughter"): "grandson",
    ("daughter", "son"): "granddaughter", ("daughter", "daughter"): "granddaughter",
}

_REL_PATTERN = re.compile(
    r"(\w+) is (\w+)'s (father|mother|son|daughter)\.\s*"
    r"(\w+) is (\w+)'s (father|mother|son|daughter)\.\s*"
    r"How is (\w+) related to (\w+)\?", re.IGNORECASE
)


def solve_blood_relation(question_text: str):
    match = _REL_PATTERN.search(question_text)
    if not match:
        return {"success": False, "message": "Could not parse this into the expected "
                                              "'X is Y's <relation>' statement format."}

    a1, b1, rel1, a2, b2, rel2, q_from, q_to = match.groups()
    rel1, rel2 = rel1.lower(), rel2.lower()

    key = (rel1, rel2)
    if key not in BLOOD_COMPOSITIONS:
        return {"success": False, "message": f"No verified composition rule for "
                                              f"'{rel1}' + '{rel2}' in the current rule set."}

    answer = BLOOD_COMPOSITIONS[key]
    explanation = (
        f"'{a1} is {b1}'s {rel1}' and '{a2} is {b2}'s {rel2}' chain together through "
        f"{b1}: {a1} is {rel1} of {b1}, and {b1} is {rel2} of {b2}. "
        f"Following this relationship two steps, {q_from} is {q_to}'s {answer}."
    )
    return {"success": True, "answer": answer, "explanation": explanation}


# ---------------------------------------------------------------------------
# Direction sense
# ---------------------------------------------------------------------------

DIR_VECTORS = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
_MOVE_PATTERN = re.compile(r"(\d+)\s*km\s+(north|south|east|west)", re.IGNORECASE)


def solve_direction_sense(question_text: str):
    moves = _MOVE_PATTERN.findall(question_text)
    if not moves:
        return {"success": False, "message": "Could not find any '<distance>km <direction>' "
                                              "movements in this question."}

    x, y = 0, 0
    steps = []
    for dist, direction in moves:
        dx, dy = DIR_VECTORS[direction.lower()]
        x += dx * int(dist)
        y += dy * int(dist)
        steps.append(f"{dist}km {direction}")

    distance = round(math.sqrt(x ** 2 + y ** 2), 2)
    explanation = (
        f"Tracking each movement as a vector: {', then '.join(steps)}. "
        f"Summing the north-south and east-west components gives a net position of "
        f"({x}, {y}) relative to the start. Straight-line distance = "
        f"sqrt({x}^2 + {y}^2) = {distance} km."
    )
    return {"success": True, "answer": distance, "explanation": explanation}


# ---------------------------------------------------------------------------
# Coding-decoding
# ---------------------------------------------------------------------------

_CODE_PATTERN = re.compile(
    r"If (\w+) is coded as (\w+), how is (\w+) coded\?", re.IGNORECASE
)


def solve_coding_decoding(question_text: str):
    match = _CODE_PATTERN.search(question_text)
    if not match:
        return {"success": False, "message": "Could not parse this into the expected "
                                              "'If WORD is coded as CODE, how is WORD2 coded?' format."}

    original, coded, new_word = match.groups()
    original, coded, new_word = original.upper(), coded.upper(), new_word.upper()

    if len(original) != len(coded):
        return {"success": False, "message": "Original and coded words have different lengths -- "
                                              "this isn't a simple per-letter shift cipher."}

    shifts = [(ord(c2) - ord(c1)) % 26 for c1, c2 in zip(original, coded)]
    if len(set(shifts)) != 1:
        return {"success": False, "message": "No constant per-letter shift found -- this may use "
                                              "a different (non-shift) coding scheme."}

    shift = shifts[0]
    new_coded = "".join(chr((ord(ch) - 65 + shift) % 26 + 65) for ch in new_word)
    explanation = (
        f"Comparing {original} -> {coded} letter by letter shows every letter shifts forward "
        f"by {shift} in the alphabet (e.g. {original[0]} -> {coded[0]}). "
        f"Applying the same +{shift} shift to each letter of {new_word} gives {new_coded}."
    )
    return {"success": True, "answer": new_coded, "explanation": explanation}


# ---------------------------------------------------------------------------
# Syllogisms
# ---------------------------------------------------------------------------

_SYLL_PATTERN = re.compile(
    r"Statement 1:\s*(.+?)\.\s*Statement 2:\s*(.+?)\.\s*Conclusion:\s*(.+?)\.\s*Is the conclusion valid\?",
    re.IGNORECASE
)

# (premise1_form, premise2_form) -> valid_conclusion_form, using A/B/C as category placeholders
VALID_SYLLOGISM_FORMS = [
    ("All {A} are {B}", "All {B} are {C}", "All {A} are {C}"),
    ("All {A} are {B}", "No {B} are {C}", "No {A} are {C}"),
    ("Some {A} are {B}", "All {B} are {C}", "Some {A} are {C}"),
    ("No {A} are {B}", "Some {B} are {C}", "Some {C} are not {A}"),
]


def _extract_categories(p1, p2):
    """Pulls out the category words from two premises sharing a middle term."""
    words1 = re.findall(r"\b[a-z]+\b", p1.lower())
    words2 = re.findall(r"\b[a-z]+\b", p2.lower())
    cats1 = [w for w in words1 if w not in ("all", "no", "some", "are", "not")]
    cats2 = [w for w in words2 if w not in ("all", "no", "some", "are", "not")]
    if len(cats1) < 2 or len(cats2) < 2:
        return None
    a, b = cats1[0], cats1[1]
    b2, c = cats2[0], cats2[1]
    if b != b2:
        return None
    return a, b, c


def solve_syllogism(question_text: str):
    match = _SYLL_PATTERN.search(question_text)
    if not match:
        return {"success": False, "message": "Could not parse this into the expected "
                                              "'Statement 1 / Statement 2 / Conclusion' format."}

    p1, p2, given_concl = [s.strip() for s in match.groups()]
    cats = _extract_categories(p1, p2)
    if cats is None:
        return {"success": False, "message": "Could not identify a shared middle term "
                                              "between the two statements."}
    a, b, c = cats

    for form_p1, form_p2, form_concl in VALID_SYLLOGISM_FORMS:
        expected_p1 = form_p1.format(A=a, B=b, C=c)
        expected_p2 = form_p2.format(A=a, B=b, C=c)
        if p1.lower() == expected_p1.lower() and p2.lower() == expected_p2.lower():
            correct_concl = form_concl.format(A=a, B=b, C=c)
            is_valid = given_concl.lower() == correct_concl.lower()
            explanation = (
                f"This matches a valid syllogism pattern: '{expected_p1}' and '{expected_p2}' "
                f"together only guarantee the conclusion '{correct_concl}'. "
                + (f"The given conclusion matches this exactly, so it IS valid."
                   if is_valid else
                   f"The given conclusion ('{given_concl}') does not match this, so it is NOT valid.")
            )
            return {"success": True, "answer": "valid" if is_valid else "invalid", "explanation": explanation}

    explanation = (
        f"The premises '{p1}' and '{p2}' do not match any of the standard valid syllogism "
        f"forms in the current rule set (e.g. two 'Some' premises never guarantee a conclusion "
        f"-- this is the classic 'undistributed middle' fallacy). Treating this as invalid."
    )
    return {"success": True, "answer": "invalid", "explanation": explanation}


# ---------------------------------------------------------------------------
# Top-level router
# ---------------------------------------------------------------------------

SOLVERS = {
    "blood_relation": solve_blood_relation,
    "direction_sense": solve_direction_sense,
    "coding_decoding": solve_coding_decoding,
    "syllogism": solve_syllogism,
}


def solve_verbal_question(question_text: str):
    """
    Classifies the question's sub-type via the trained classifier, then
    dispatches to the matching solver. Returns the solver's result plus
    the detected subtype for transparency.
    """
    subtype = classify_subtype(question_text)
    solver = SOLVERS.get(subtype)
    if solver is None:
        return {"success": False, "message": f"No solver registered for subtype '{subtype}'."}

    result = solver(question_text)
    result["detected_subtype"] = subtype
    return result


if __name__ == "__main__":
    tests = [
        "Rahul is Priya's father. Priya is Ankit's mother. How is Rahul related to Ankit?",
        "Sonia walks 4km north, then 6km east. How far is Sonia from the starting point?",
        "If DOG is coded as EPH, how is CAT coded?",
        "Statement 1: All roses are flowers. Statement 2: All flowers are plants. "
        "Conclusion: All roses are plants. Is the conclusion valid?",
        "Statement 1: Some cats are dogs. Statement 2: Some dogs are birds. "
        "Conclusion: Some cats are birds. Is the conclusion valid?",
    ]
    for t in tests:
        print(solve_verbal_question(t))
        print()

"""
concept_map.py

Maps each rule_id from generate_dataset.py to a human-readable concept
description. These descriptions are what actually get embedded by
Sentence-BERT in weakness_diagnoser.py -- the embedding model has no idea
what "T3" means, but it can meaningfully compare "product of two numbers
minus a third" against "product and subtraction relationship in a grid"
and recognize they're conceptually similar.
"""

CONCEPT_DESCRIPTIONS = {
    # Triangle rules
    "T1": "sum of three outer numbers gives the center value",
    "T2": "sum of three outer numbers doubled gives the center value",
    "T3": "product of two numbers minus a third number",
    "T4": "sum of two numbers minus a third number",
    "T5": "sum of two numbers minus twice a third number",
    "T6": "product of two numbers minus a third number, different ordering",
    "T7": "combination of addition, multiplication and halving",
    "T8": "square of one number minus the sum of the other two",
    "T9": "average of three numbers scaled by a constant factor",
    "T10": "combination of addition, multiplication and subtraction of three numbers",
    # Grid rules
    "G1": "sum of three grid values gives the missing value",
    "G2": "sum of two grid values minus a third grid value",
    "G3": "product and division relationship between grid values",
    "G4": "sum and difference relationship between grid values",
    "G5": "product and subtraction relationship between grid values",
    # Series rules
    "S1": "arithmetic progression with a constant difference between terms",
    "S2": "geometric progression with a constant ratio between terms",
    "S3": "quadratic pattern based on squares of position",
    "S4": "alternating addition and subtraction pattern depending on position",
    "S5": "progressively increasing difference between consecutive terms",
}


def describe(rule_id: str) -> str:
    return CONCEPT_DESCRIPTIONS.get(rule_id, rule_id)

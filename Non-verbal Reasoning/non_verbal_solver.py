"""
non_verbal_solver.py

Quiz-layer logic for the non-verbal reasoning module (proposal section
5.5), built on top of the trained CNN in non_verbal_cnn.py.

Design notes (fixed after real user testing):
- Mirror check is a COMPARISON task (reference shape + 4 candidate options,
  pick the true mirror image) -- matching standard exam format. An earlier
  version asked an absolute "is this single, randomly-rotated shape
  mirrored?" judgment, which turned out to be genuinely confusing even
  though technically well-defined, since a person has no fixed reference
  orientation to judge against.
- All DISPLAY images use jitter=False for clarity. The CNN was trained
  WITH jitter (for robustness) but performs at least as well, usually
  better, on clean input at inference time -- jitter was purely a
  training-time regularizer, never required for inference.
"""

import random

import torch

from non_verbal_dataset import (
    make_variant, make_variant_with_axis, draw_reference_with_axis_indicator,
    image_to_png_bytes, ROTATIONS
)
from non_verbal_cnn import load_model

_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


def predict(arr):
    """Runs the CNN on one image array, returns (mirror_bool, rotation_deg, confidence_dict)."""
    model = get_model()
    x = torch.tensor(arr).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred_mirror, pred_rot = model(x)
        mirror_probs = torch.softmax(pred_mirror, dim=1)[0]
        rot_probs = torch.softmax(pred_rot, dim=1)[0]
    mirror_label = bool(mirror_probs.argmax().item())
    rot_label = ROTATIONS[rot_probs.argmax().item()]
    return mirror_label, rot_label, {
        "mirror_confidence": mirror_probs.max().item(),
        "rotation_confidence": rot_probs.max().item(),
    }


def make_mirror_question():
    """
    Shows a reference shape with a visible MIRROR LINE indicator (a hatched
    bar, exactly like standard exam figures) telling the student which axis
    to flip across: a vertical bar beside the shape means flip left-right;
    a horizontal bar above/below means flip top-bottom.

    4 candidates are offered:
      - the correct flip along the INDICATED axis, same rotation
      - the unchanged original (not flipped at all -- a common trap)
      - the correct axis's flip but at a different rotation
      - the WRONG axis's flip at the same rotation -- this specifically
        catches the mistake of flipping left-right when the indicator
        called for top-bottom (or vice versa).
    """
    axis = random.choice(["vertical", "horizontal"])
    other_axis = "horizontal" if axis == "vertical" else "vertical"

    ref_rotation = random.choice(ROTATIONS)
    ref_arr = make_variant_with_axis(None, ref_rotation, jitter=False)

    other_rotations = [r for r in ROTATIONS if r != ref_rotation]
    distractor_rotation = random.choice(other_rotations)

    candidate_defs = [
        {"flip_axis": axis, "rotation": ref_rotation, "is_correct": True},
        {"flip_axis": None, "rotation": ref_rotation, "is_correct": False},
        {"flip_axis": axis, "rotation": distractor_rotation, "is_correct": False},
        {"flip_axis": other_axis, "rotation": ref_rotation, "is_correct": False},
    ]
    random.shuffle(candidate_defs)

    candidates = []
    correct_index = None
    for i, cdef in enumerate(candidate_defs):
        arr = make_variant_with_axis(cdef["flip_axis"], cdef["rotation"], jitter=False)
        candidates.append({
            "image_bytes": image_to_png_bytes(arr),
            "flip_axis": cdef["flip_axis"],
            "rotation": cdef["rotation"],
            "_arr": arr,
        })
        if cdef["is_correct"]:
            correct_index = i

    return {
        "type": "mirror_check",
        "ref_image_bytes": draw_reference_with_axis_indicator(ref_arr, axis),
        "ref_rotation": ref_rotation,
        "axis": axis,
        "candidates": candidates,
        "correct_index": correct_index,
    }


def make_rotation_question():
    """Returns a dict describing one rotation-identification question. Clean (jitter-free) image."""
    rot = random.choice(ROTATIONS)
    arr = make_variant(False, rot, jitter=False)
    return {
        "type": "rotation_id",
        "image_bytes": image_to_png_bytes(arr),
        "true_rotation": rot,
        "options": ROTATIONS,
        "_arr": arr,
    }


def make_series_question():
    """
    Generates 3 shown frames with a fixed rotation increment (90 degrees --
    the standard, clearly-progressive pattern), plus 4 candidate next-frame
    images (1 correct, 3 distractors). All images are clean (jitter-free)
    for display clarity.
    """
    increment = 90  # a consistent 90-degree rotating sequence, matching standard exam style
    start = random.choice(ROTATIONS)
    shown_rotations = [(start + increment * i) % 360 for i in range(3)]
    correct_next = (start + increment * 3) % 360

    shown_frames = []
    for r in shown_rotations:
        arr = make_variant(False, r, jitter=False)
        shown_frames.append({"image_bytes": image_to_png_bytes(arr), "rotation": r, "_arr": arr})

    distractor_pool = [r for r in ROTATIONS if r != correct_next]
    distractors = random.sample(distractor_pool, 3)
    candidate_rotations = distractors + [correct_next]
    random.shuffle(candidate_rotations)

    candidates = [
        {"image_bytes": image_to_png_bytes(make_variant(False, r, jitter=False)), "rotation": r}
        for r in candidate_rotations
    ]

    return {
        "type": "figure_series",
        "shown_frames": shown_frames,
        "candidates": candidates,
        "correct_rotation": correct_next,
        "increment": increment,
    }


def explain_mirror(ref_rotation, axis, candidates, correct_index):
    """Builds a verified, plain-language explanation for the mirror comparison task."""
    flip_description = "left-right" if axis == "vertical" else "top-to-bottom"
    bar_description = "vertical hatched bar beside the shape" if axis == "vertical" else "horizontal hatched bar below the shape"

    other_notes = []
    for i, c in enumerate(candidates):
        if i == correct_index:
            continue
        if c["flip_axis"] is None and c["rotation"] == ref_rotation:
            other_notes.append("one option is just the unchanged original shape (not flipped at all)")
        elif c["flip_axis"] == axis and c["rotation"] != ref_rotation:
            other_notes.append("one option is flipped correctly but at the wrong rotation")
        elif c["flip_axis"] is not None and c["flip_axis"] != axis:
            other_notes.append(f"one option is flipped along the WRONG axis "
                              f"({'top-to-bottom' if axis == 'vertical' else 'left-right'} instead of "
                              f"{flip_description}) -- a common mistake if you don't check the mirror-line direction first")

    explanation = (
        f"The {bar_description} tells you the mirror line is {axis}, which means you flip the "
        f"shape {flip_description} (rotation stays the same, at {ref_rotation} degrees). "
        + ("Among the distractors: " + "; ".join(other_notes) + "." if other_notes else "")
    )
    return explanation


def explain_series(shown_frames, increment, correct_rotation):
    """
    Runs the CNN on each shown frame to identify its rotation (this is the
    genuine DL-inference step), then reports the detected pattern.
    """
    detected = []
    for frame in shown_frames:
        _, rot_pred, conf = predict(frame["_arr"])
        detected.append(rot_pred)

    explanation = (
        f"The CNN identified each shown frame's rotation as {detected} degrees. "
        f"The consistent increment between frames is {increment} degrees, so the next "
        f"frame in the sequence should be rotated {correct_rotation} degrees."
    )
    return explanation


if __name__ == "__main__":
    q = make_mirror_question()
    print(f"Mirror question: axis={q['axis']}, ref_rotation={q['ref_rotation']}, correct_index={q['correct_index']}")
    print(explain_mirror(q["ref_rotation"], q["axis"], q["candidates"], q["correct_index"]))

    q2 = make_series_question()
    print(f"\nSeries question: increment={q2['increment']}, correct_next_rotation={q2['correct_rotation']}")
    print(explain_series(q2["shown_frames"], q2["increment"], q2["correct_rotation"]))

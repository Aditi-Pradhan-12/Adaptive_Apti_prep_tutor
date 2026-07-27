"""
non_verbal_dataset.py

Generates synthetic shape images for the non-verbal reasoning CNN
(proposal section 5.5): mirror image recognition, rotation recognition,
and (built on top of rotation recognition) figure-series completion.

Uses a deliberately ASYMMETRIC arrow-like shape -- if the shape were
symmetric (like a plain circle or square), mirroring/rotating it wouldn't
change anything visually, and the CNN would have nothing to learn. An
asymmetric shape means all 4 rotations x 2 mirror states = 8 distinct
appearances, which is exactly what makes this a genuine, non-trivial
classification problem.
"""

import io
import random

import numpy as np
from PIL import Image, ImageDraw

IMG_SIZE = 32
ROTATIONS = [0, 90, 180, 270]  # class indices 0,1,2,3


def draw_base_shape(size: int = IMG_SIZE) -> Image.Image:
    """An asymmetric arrow/flag shape -- distinct under every rotation and mirroring."""
    img = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(img)
    draw.polygon(
        [(4, 4), (4, 22), (14, 22), (14, 28), (28, 14), (14, 4)],
        fill=0,
    )
    return img


def make_variant(mirror: bool, rotation_deg: int, jitter: bool = True) -> np.ndarray:
    """
    Produces one image variant: base shape -> optional mirror -> rotate by
    a multiple of 90 (exact multiples avoid interpolation blur/artifacts).
    Returns a normalized (0-1) float32 numpy array, shape (IMG_SIZE, IMG_SIZE).
    """
    img = draw_base_shape()
    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.rotate(-rotation_deg, fillcolor=255)  # PIL rotates counter-clockwise; negate for clockwise convention

    arr = np.array(img, dtype=np.float32) / 255.0

    if jitter:
        # Small random shift + noise so the CNN can't just memorize exact pixel
        # positions -- forces it to genuinely learn shape/orientation features.
        shift_x, shift_y = random.randint(-2, 2), random.randint(-2, 2)
        arr = np.roll(arr, shift_x, axis=1)
        arr = np.roll(arr, shift_y, axis=0)
        arr = arr + np.random.normal(0, 0.02, arr.shape).astype(np.float32)
        arr = np.clip(arr, 0, 1)

    return arr


def make_variant_with_axis(flip_axis, rotation_deg: int, jitter: bool = False) -> np.ndarray:
    """
    Like make_variant(), but explicit about WHICH axis the mirror line sits
    on -- flip_axis is one of: None (no flip), "vertical" (mirror line is a
    vertical bar beside the shape -> flip left-right), or "horizontal"
    (mirror line is a horizontal bar above/below the shape -> flip top-bottom).

    Kept as a separate function from make_variant() so the CNN's training
    data pipeline (which only ever used left-right flips) is completely
    unaffected -- this is purely for the mirror-comparison quiz, which
    grades against known ground truth rather than a CNN prediction anyway.
    """
    img = draw_base_shape()
    if flip_axis == "vertical":
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif flip_axis == "horizontal":
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    img = img.rotate(-rotation_deg, fillcolor=255)

    arr = np.array(img, dtype=np.float32) / 255.0
    if jitter:
        shift_x, shift_y = random.randint(-2, 2), random.randint(-2, 2)
        arr = np.roll(arr, shift_x, axis=1)
        arr = np.roll(arr, shift_y, axis=0)
        arr = arr + np.random.normal(0, 0.02, arr.shape).astype(np.float32)
        arr = np.clip(arr, 0, 1)
    return arr


def draw_reference_with_axis_indicator(arr: np.ndarray, axis: str) -> bytes:
    """
    Composites the shape image with a hatched mirror-line bar next to it,
    exactly like standard exam figures (a hatched vertical bar beside the
    shape means 'mirror left-right'; a hatched horizontal bar above/below
    means 'mirror top-bottom'). Returns PNG bytes ready for st.image().
    """
    shape_img = Image.fromarray((arr * 255).astype(np.uint8), mode="L").convert("RGB")
    bar_thickness = 10
    gap = 6

    if axis == "vertical":
        canvas = Image.new("RGB", (IMG_SIZE + gap + bar_thickness, IMG_SIZE), color="white")
        canvas.paste(shape_img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        bar_x0 = IMG_SIZE + gap
        bar_x1 = bar_x0 + bar_thickness
        draw.rectangle([bar_x0, 0, bar_x1, IMG_SIZE], outline="black")
        for y in range(-IMG_SIZE, IMG_SIZE, 5):
            draw.line([(bar_x0, y), (bar_x1, y + IMG_SIZE)], fill="black", width=1)
    else:  # horizontal
        canvas = Image.new("RGB", (IMG_SIZE, IMG_SIZE + gap + bar_thickness), color="white")
        canvas.paste(shape_img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        bar_y0 = IMG_SIZE + gap
        bar_y1 = bar_y0 + bar_thickness
        draw.rectangle([0, bar_y0, IMG_SIZE, bar_y1], outline="black")
        for x in range(-IMG_SIZE, IMG_SIZE, 5):
            draw.line([(x, bar_y0), (x + IMG_SIZE, bar_y1)], fill="black", width=1)

    canvas = canvas.resize((canvas.width * 5, canvas.height * 5), Image.NEAREST)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def generate_dataset(n_samples: int = 6000, jitter: bool = True):
    """
    Returns (images, mirror_labels, rotation_labels) as numpy arrays, ready
    for training. mirror_labels: 0=normal, 1=mirrored. rotation_labels: 0-3
    corresponding to ROTATIONS index.
    """
    images = np.zeros((n_samples, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    mirror_labels = np.zeros(n_samples, dtype=np.int64)
    rotation_labels = np.zeros(n_samples, dtype=np.int64)

    for i in range(n_samples):
        mirror = random.choice([0, 1])
        rot_idx = random.choice([0, 1, 2, 3])
        images[i] = make_variant(bool(mirror), ROTATIONS[rot_idx], jitter=jitter)
        mirror_labels[i] = mirror
        rotation_labels[i] = rot_idx

    return images, mirror_labels, rotation_labels


def image_to_png_bytes(arr: np.ndarray) -> bytes:
    """Converts a (IMG_SIZE, IMG_SIZE) float array back to PNG bytes for display in Streamlit."""
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    img = img.resize((160, 160), Image.NEAREST)  # upscale for visibility in the UI
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    imgs, mirrors, rots = generate_dataset(10, jitter=False)
    print("Generated shapes:", imgs.shape, "mirror labels:", mirrors, "rotation labels:", rots)
